"""Serial, durable, read-only paper Agent execution; scoring comes later."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

from ..agent import Agent
from ..bundle import prompt_provider, tools_provider
from ..capabilities import settings_provider
from ..cordis import Context
from ..persistence import JsonlStore
from ..policies import meter_provider, retry_plugin, checkpoint_plugin, pruner_plugin, spill_plugin
from ..session import Session
from .dataset import bind_papers
from .trace import TraceStore, redact


class EvaluationStopped(RuntimeError):
    code = "EVALUATION_STOPPED"


class EvaluationWorker:
    """One worker per database. A shared PaperRuntime is owned by the caller.

    The client factory is trusted application configuration, never dataset code.
    Network clients must use bounded request timeouts (DeepSeekClient: 60s).
    Cancellation/deadline is cooperative; a running synchronous call is not killed.
    """
    def __init__(self, store, client_factory, runtime=None, *, allow_judge_api=False):
        self.store = store
        self.client_factory = client_factory
        self.runtime = runtime
        self.allow_judge_api = allow_judge_api
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Worker already running")
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._background, name="agent-evaluation", daemon=False)
        self._thread.start()

    def _background(self):
        try:
            self.work(wait=True)
        except Exception as error:
            self.error = error

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def work(self, *, wait=False, run_id=None):
        with self.store.worker_lock():
            self.store.recover()
            while not self._stop.is_set():
                claimed = self.store.claim(run_id)
                if claimed is None:
                    if not wait:
                        return
                    self._stop.wait(0.2)
                    continue
                try:
                    self._run(claimed)
                except Exception as error:
                    self.store.finish_run(claimed, "failed", type(error).__name__)
                if run_id is not None:
                    return

    def _bindings(self, dataset):
        if dataset.papers and self.runtime is None:
            raise ValueError("Dataset requires a PaperRuntime")
        return bind_papers(dataset, self.runtime.catalog if self.runtime else None)

    def _run(self, run_id):
        configuration = json.loads(self.store.get_run(run_id)["configuration"])
        judge_config = configuration.get("judge") if self.store.get_run(run_id)["mode"] == "live_judge" else None
        if judge_config and not self.allow_judge_api:
            raise ValueError("This run requires explicit judge API authorization")
        catalog = configuration.get("catalog")
        if catalog is not None and (self.runtime is None or Path(self.runtime.config.catalog_path).resolve() != Path(catalog).resolve()):
            raise ValueError("Worker catalog differs from the queued configuration")
        dataset = self.store.dataset(run_id)
        self._bindings(dataset)  # Fail before any model call on stale corpus.
        completed = {a["case_id"] for a in self.store.attempts(run_id) if a["status"] == "completed"}
        failed = False
        for case in dataset.cases:
            if self._stop.is_set() or self.store.get_run(run_id)["status"] == "cancelling":
                self.store.finish_run(run_id, "interrupted", "worker_stopped")
                return
            if case.case_id in completed:
                continue
            status = self._case(run_id, dataset, case)
            if status == "completed" and judge_config:
                from .scoring import SemanticScorer
                attempt = self.store.attempts(run_id)[-1]
                SemanticScorer(self.store, judge_config, allow_api=True).score(attempt,
                    cancelled=lambda: self._stop.is_set() or self.store.get_run(run_id)["status"] == "cancelling")
            if status == "interrupted":
                self.store.finish_run(run_id, "interrupted", "worker_stopped")
                return
            failed |= status == "failed"
        self.store.finish_run(run_id, "failed" if failed else "completed")

    def _case(self, run_id, dataset, case):
        attempt_id, session_id, directory = self.store.start_attempt(run_id, case.case_id)
        session = Session()
        ctx = Context()
        status, error_type, error_message = "completed", None, None
        started = time.monotonic()
        traces = TraceStore(directory / "artifacts")
        journal = JsonlStore(directory / "trace.jsonl")
        used_tokens = 0
        missing_usage = False

        def flush(current):
            # Never persist hidden reasoning or credentials in the evaluation journal.
            safe = Session.from_log([replace(e, data=redact(dict(e.data), traces.secrets)) for e in current.events])
            journal.save(safe)

        def boundary(*_):
            if self._stop.is_set() or self.store.get_run(run_id)["status"] == "cancelling":
                raise EvaluationStopped("cancelled or worker stopped")
            if time.monotonic() - started >= case.budgets.timeout_seconds:
                raise TimeoutError("Case deadline exceeded at a safe boundary")
            # Cumulative usage is telemetry, not a multi-turn failure condition.
            self._bindings(dataset)  # Detect active-version changes during execution.

        try:
            raw_client = self.client_factory()
            traces = TraceStore(directory / "artifacts", secrets=(getattr(raw_client, "api_key", None),))

            class BudgetClient:
                MODEL = raw_client.MODEL
                def chat(inner, messages, tools):
                    nonlocal used_tokens, missing_usage
                    boundary()
                    reply = raw_client.chat(messages, tools)
                    usage = reply.usage or {}
                    total = usage.get("total_tokens")
                    if type(total) is not int:
                        total = (usage["prompt_tokens"] + usage["completion_tokens"]
                                 if "prompt_tokens" in usage and "completion_tokens" in usage else None)
                    if total is None:
                        missing_usage = True
                    else:
                        used_tokens += total
                    return reply

            ctx.plugin(settings_provider, {})
            ctx.plugin(prompt_provider)
            ctx.plugin(tools_provider)
            ctx.plugin(meter_provider)
            ctx.plugin(retry_plugin)
            ctx.plugin(checkpoint_plugin, flush)
            ctx.plugin(pruner_plugin)
            ctx.plugin(spill_plugin, str(directory / "spills"))
            if self.runtime is not None:
                from ..rag.plugin import rag_consumer
                ctx.provide("paper_runtime", self.runtime)
                ctx.plugin(rag_consumer)
            agent = Agent(ctx, BudgetClient(), ctx.require("tools"), ctx.require("prompt"),
                          variables={"name": "小知"}, session=session, session_id=session_id,
                          trace_store=traces, max_steps_per_turn=case.budgets.max_steps_per_turn,
                          trace_identity=dict(run_id=run_id, case_id=case.case_id, attempt_id=attempt_id))
            ctx.on("agent/pre-step", boundary)
            ctx.on("tools/pre-execute", boundary)
            session.append("evaluation/start", dict(run_id=run_id, case_id=case.case_id,
                           attempt_id=attempt_id, session_id=session_id,
                           profile="paper-readonly-v1", model=raw_client.MODEL))
            for number, turn in enumerate(case.turns, 1):
                boundary()
                mapping = self._bindings(dataset)
                selected = [mapping[key] for key in turn.selected_papers]
                if self.runtime is not None:
                    ctx.require("paper_scope")["paper_ids"] = tuple(selected)
                session.append("rag/scope", {"paper_ids": selected})
                agent.followup(turn.input)  # Sole dataset-to-model input projection.
                agent.run(max_turns=number)
                flush(session)
                boundary()
        except EvaluationStopped as error:
            status, error_type, error_message = "interrupted", type(error).__name__, str(error)
        except Exception as error:
            status, error_type, error_message = "failed", type(error).__name__, str(error)
        finally:
            ctx.dispose()
        session.append("evaluation/end", dict(status=status, error_type=error_type))
        flush(session)
        # Scoring errors are separate from execution errors and never replay model calls.
        from .rules import score_case
        scoring_status, scoring_error = "rules_completed", None
        metrics = []
        try:
            metrics = score_case(case, dataset, session.events, traces.root, target_id=attempt_id)
            self.store.save_metrics(attempt_id, metrics)
        except Exception as error:
            scoring_status, scoring_error = "error", type(error).__name__
        report = traces.put(dict(run_id=run_id, case_id=case.case_id, attempt_id=attempt_id,
                                 session_id=session_id, status=status, error_type=error_type, error_message=error_message,
                                 verdict="failed" if any(m.passed is False for m in metrics) else "incomplete", scoring_status=scoring_status, scoring_error=scoring_error,
                                 metrics=[m.model_dump(mode="json") for m in metrics],
                                 elapsed_ms=(time.monotonic()-started)*1000,
                                 total_tokens=used_tokens, usage_complete=not missing_usage,
                                 answers=[e.data.get("content") for e in session.events if e.type == "assistant/message"],
                                 trace="trace.jsonl"))
        self.store.finish_attempt(attempt_id, status, report, error_type)
        return status
