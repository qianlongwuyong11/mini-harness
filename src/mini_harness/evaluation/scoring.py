"""Main-process scoring coordinator; never imports DeepEval."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4
from urllib.parse import urlparse
from pydantic import Field, model_validator
from .models import Contract, MetricResult
from .dataset import fingerprint
from .metrics import read_artifact
from .registry import validate_metric_result
from .trace import TraceStore, redact
from .backend import worker_environment

SUPPORTED = {"task_success", "answer_relevance", "answer_correctness", "requirement_coverage", "faithfulness", "tool_correctness"}


class JudgeConfig(Contract):
    model: str = Field(min_length=1)
    base_url: str
    api_key_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    metrics: list[str] = Field(default_factory=lambda: ["task_success", "answer_relevance", "answer_correctness", "requirement_coverage", "faithfulness", "tool_correctness"])
    thresholds: dict[str, float] = Field(default_factory=dict)
    request_timeout: int = Field(default=45, ge=1, le=120)
    process_timeout: int = Field(default=180, ge=1, le=1800)
    max_calls: int = Field(default=12, ge=1, le=100)
    max_output_tokens: int = Field(default=2048, ge=128, le=8192)

    @model_validator(mode="after")
    def valid(self):
        url = urlparse(self.base_url)
        if (url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1"))) or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Judge endpoint must be HTTPS or loopback HTTP without credentials/query")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics) or not set(self.metrics) <= SUPPORTED:
            raise ValueError("Unsupported/duplicate/empty judge metric list")
        if not set(self.thresholds) <= set(self.metrics) or any(not 0 <= v <= 1 for v in self.thresholds.values()):
            raise ValueError("Thresholds must be explicitly configured in [0,1] for selected metrics")
        return self


def prepare_payload(store, attempt):
    from ..persistence import JsonlStore
    if attempt["status"] != "completed":
        raise ValueError("Only completed Agent attempts can be semantically scored")
    dataset = store.dataset(attempt["run_id"])
    case = next(c for c in dataset.cases if c.case_id == attempt["case_id"])
    bindings = {p.logical_key: p for p in dataset.papers}
    directory = (store.root / attempt["artifact_dir"]).resolve()
    if not directory.is_relative_to(store.root):
        raise ValueError("Attempt directory outside evaluation store")
    journal = (directory / "trace.jsonl").resolve(strict=True)
    if not journal.is_relative_to(directory) or not (directory / "artifacts").resolve().is_relative_to(directory):
        raise ValueError("Trace path escapes attempt directory")
    session = JsonlStore(journal).load()
    groups = []
    for event in session.events:
        if event.type == "turn/start":
            groups.append([])
        if groups:
            groups[-1].append(event)
    if len(groups) != len(case.turns):
        raise ValueError("Conversation trace is incomplete")
    turns, names = [], []
    for source, events in zip(case.turns, groups):
        if not any(e.type == "turn/end" and e.data.get("reason") == "completed" for e in events):
            raise ValueError("Turn trace is incomplete")
        finals = [e for e in events if e.type == "assistant/message" and not e.data.get("tool_calls")]
        if not finals:
            raise ValueError("Final answer missing")
        final = finals[-1]
        evidence = []
        for event in events:
            if event.type == "tool/call":
                names.append(event.data["name"])
            if event.type == "rag/context" and event.data.get("model_request_id") == final.data.get("model_request_id"):
                snapshot = read_artifact(directory / "artifacts", event.data["input_ref"])
                for message in snapshot["messages"]:
                    content = message.get("content") or ""
                    if content.startswith("本轮证据快照"):
                        evidence = [json.dumps(item, ensure_ascii=False) for item in json.loads(content.split("\n", 1)[1])]
        turns.append(dict(input=source.input, selected_papers=[{"paper_id": bindings[key].paper_id, "version": bindings[key].version} for key in source.selected_papers],
                          output=final.data.get("content") or "", evidence=evidence))
    return dict(attempt_id=attempt["attempt_id"], turns=turns, tool_names=list(dict.fromkeys(names)),
                case={key: case.model_dump(mode="json")[key] for key in (
                    "expected_behavior", "reference_answer", "key_points", "answerable", "tool_constraints")})


class SemanticScorer:
    def __init__(self, store, config, *, allow_api=False):
        self.store = store
        self.config = config if isinstance(config, JudgeConfig) else JudgeConfig.model_validate(config)
        self.allow_api = allow_api

    def score(self, attempt, *, cancelled=lambda: False):
        if not self.allow_api:
            raise ValueError("Explicit judge API authorization required")
        score_id = uuid4().hex
        config = self.config.model_dump(mode="json")
        self.store.start_score(score_id, attempt["attempt_id"], config)
        artifacts = TraceStore(self.store.root / attempt["artifact_dir"] / "scores" / score_id,
                               secrets=(os.environ.get(config["api_key_env"]),))
        report = {"score_id": score_id, "attempt_id": attempt["attempt_id"], "config": config,
                  "configuration_sha256": fingerprint(config), "verdict": "incomplete"}
        process = None
        try:
            payload = prepare_payload(self.store, attempt)
            request = dict(input=payload, config=config, artifact_root=str(artifacts.root))
            request = redact(request, artifacts.secrets)
            # Config is application control data, not text to redact as credentials.
            request["config"] = config
            raw = json.dumps(request, ensure_ascii=False)
            if len(raw.encode("utf-8")) > 2 * 1024 * 1024:
                raise ValueError("Judge input exceeds 2 MiB; no silent truncation")
            report["input_ref"] = artifacts.put(payload)
            repo = Path(__file__).resolve().parents[3]
            python = repo / ".venv-eval" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not python.is_file():
                raise FileNotFoundError("Install the isolated .venv-eval environment first")
            env = worker_environment()
            env["PYTHONPATH"] = str(repo / "src")
            env["PYTHONIOENCODING"] = "utf-8"
            started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="mini-harness-judge-") as cwd:
                process = subprocess.Popen([str(python), "-m", "mini_harness.evaluation.judge_worker"],
                    cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                first = True
                while True:
                    if cancelled():
                        raise InterruptedError("Judge cancelled")
                    if time.monotonic() - started > config["process_timeout"]:
                        raise TimeoutError("Judge process deadline exceeded")
                    try:
                        stdout, stderr = process.communicate(input=raw if first else None, timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        first = False
                if process.returncode:
                    report["diagnostic_ref"] = artifacts.put({"stdout": stdout, "stderr": stderr})
                    report["error_summary"] = redact(stdout + stderr, artifacts.secrets)[:2000]
                    raise RuntimeError("Isolated judge process failed")
            child = json.loads(stdout)
            metrics = [validate_metric_result(MetricResult.model_validate(row)) for row in child["metrics"]]
            if {m.metric_id for m in metrics} != set(config["metrics"]) or any(m.target_id != attempt["attempt_id"] for m in metrics):
                raise ValueError("Unexpected judge metric identity")
            report.update(child)
            calls = child["calls"]
            report["usage"] = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens"):
                values = [(call.get("usage") or {}).get(key) for call in calls]
                report["usage"][key] = sum(values) if values and all(type(v) is int and v >= 0 for v in values) else None
            report["cost"] = None  # Requires a versioned model price table.
            report["request_count"] = len(calls)
            report["status"] = "error" if any(m.status == "error" for m in metrics) else "completed"
            report["semantic_verdict"] = "failed" if any(m.passed is False for m in metrics) else (
                "passed" if metrics and all(m.passed is True for m in metrics) else "incomplete")
            rules = self.store.metrics(attempt["attempt_id"])
            hard = [m for m in rules if m["metric_id"] in ("scope_violations", "invalid_citations", "tool_constraints_satisfied")]
            report["verdict"] = "failed" if any(m.get("passed") is False for m in hard) else (
                report["semantic_verdict"] if len(hard) == 3 and all(m.get("passed") is True or m["status"] == "not_applicable" for m in hard) else "incomplete")
            report["elapsed_ms"] = (time.monotonic() - started) * 1000
        except Exception as error:
            report.update(status="interrupted" if isinstance(error, InterruptedError) else "error", error_type=type(error).__name__)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
        try:
            ref = artifacts.put(report)
        except Exception as error:
            report.update(status="error", error_type=type(error).__name__)
            ref = None
        self.store.finish_score(score_id, report["status"], ref, report.get("error_type"))
        return report
