import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from mini_harness.client import Message
from mini_harness.evaluation.models import EvaluationDataset
from mini_harness.evaluation.runner import EvaluationWorker
from mini_harness.evaluation.store import EvaluationStore
from mini_harness.evaluation.scoring import JudgeConfig, SemanticScorer, prepare_payload
from mini_harness.evaluation.metrics import read_artifact


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = EvaluationStore(Path(self.tmp.name) / "evals")
        self.requests = []
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps(dict(id="fixture", choices=[dict(message=dict(content=json.dumps(dict(score=8, reason="Offline fixture reason"))))],
                                       usage=dict(prompt_tokens=20, completion_tokens=8, total_tokens=28))).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.config = JudgeConfig(model="fixture-local", base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
                                  api_key_env="TEST_JUDGE_API_KEY", metrics=["task_success", "answer_relevance"],
                                  thresholds={"task_success": .7, "answer_relevance": .7})

    def attempt(self, turns=1, required_tools=None, judge_config=None):
        data = EvaluationDataset.model_validate(dict(dataset_id="fixture", cases=[dict(case_id="c",category="basic",split="regression",scenario="example",
            expected_behavior="Answer the question", tool_constraints={"required_tools": required_tools or []}, turns=[dict(input=f"q{i}",selected_papers=[]) for i in range(turns)])]))
        class Client:
            MODEL = "agent-fixture"
            def chat(self, messages, tools):
                return Message(role="assistant", content="An answer", usage={"total_tokens": 3})
        run = self.store.enqueue(data, {"judge": judge_config} if judge_config else {}, mode="live_judge" if judge_config else "live_rules")
        EvaluationWorker(self.store, Client, allow_judge_api=bool(judge_config)).work()
        return self.store.attempts(run)[0]

    def test_single_multi_actual_isolated_sdk_and_rescore(self):
        for turns, adapter in ((1, "GEval"), (2, "ConversationalGEval")):
            attempt = self.attempt(turns)
            trace = self.store.root / attempt["artifact_dir"] / "trace.jsonl"
            before = trace.read_bytes()
            with patch.dict(os.environ, {"TEST_JUDGE_API_KEY": "fixture-secret-never-store"}), self.store.worker_lock():
                scorer = SemanticScorer(self.store, self.config, allow_api=True)
                first = scorer.score(attempt)
                self.assertEqual(first["status"], "completed", first)
                self.assertEqual(first["adapter"], adapter)
                self.assertEqual(first["metrics"][0]["value"], .8)
                self.assertEqual(first["calls"][0]["usage"]["total_tokens"], 28)
                self.assertEqual(first["usage"]["total_tokens"], 56)
                self.assertEqual(first["calls"][0]["metric_id"], "task_success")
                second = scorer.score(attempt)
            self.assertNotEqual(first["score_id"], second["score_id"])
            self.assertEqual(len(self.store.scores(attempt["attempt_id"])), 2)
            self.assertEqual(trace.read_bytes(), before)
            for path in trace.parent.glob("scores/**/*.json"):
                self.assertNotIn("fixture-secret-never-store", path.read_text(encoding="utf-8"))
        self.assertEqual(len(self.requests), 8)

    def test_config_authorization_and_failed_score_preserve_agent(self):
        with self.assertRaises(ValueError):
            JudgeConfig(model="x", base_url="http://remote.example", api_key_env="K")
        with self.assertRaises(ValueError):
            JudgeConfig(**{**self.config.model_dump(), "metrics": ["unknown"]})
        attempt = self.attempt()
        with self.assertRaises(ValueError):
            SemanticScorer(self.store, self.config).score(attempt)
        with patch.dict(os.environ, {"TEST_JUDGE_API_KEY": ""}), self.store.worker_lock():
            failed = SemanticScorer(self.store, self.config, allow_api=True).score(attempt)
        self.assertEqual(failed["status"], "error")
        self.assertEqual(self.store.attempts(attempt["run_id"])[0]["status"], "completed")
        self.assertFalse(self.requests)

    def test_payload_uses_actual_final_answer(self):
        attempt = self.attempt(2)
        payload = prepare_payload(self.store, attempt)
        self.assertEqual([t["output"] for t in payload["turns"]], ["An answer", "An answer"])
        self.assertEqual(payload["case"]["expected_behavior"], "Answer the question")

    def test_payload_evidence_matches_final_model_request(self):
        from mini_harness.evaluation.trace import TraceStore
        from mini_harness.persistence import JsonlStore
        from mini_harness.session import Session
        attempt = self.attempt()
        directory = self.store.root / attempt["artifact_dir"]
        artifacts = TraceStore(directory / "artifacts")
        stale = artifacts.put({"messages": [{"content": '本轮证据快照\n[{"excerpt":"stale"}]'}]})
        final = artifacts.put({"messages": [{"content": '本轮证据快照\n[{"excerpt":"actually injected"}]'}]})
        session = Session()
        session.append("turn/start", {"turn": 1})
        session.append("rag/context", {"model_request_id": "first", "input_ref": stale})
        session.append("rag/context", {"model_request_id": "last", "input_ref": final})
        session.append("assistant/message", {"model_request_id": "last", "content": "answer", "tool_calls": []})
        session.append("turn/end", {"turn": 1, "reason": "completed"})
        # Replace only this test's disposable fixture journal.
        (directory / "trace.jsonl").unlink()
        JsonlStore(directory / "trace.jsonl").save(session)
        payload = prepare_payload(self.store, attempt)
        self.assertIn("actually injected", payload["turns"][0]["evidence"][0])
        self.assertNotIn("stale", str(payload["turns"][0]["evidence"]))

    def test_tool_metric_and_missing_references(self):
        attempt = self.attempt(required_tools=["list_papers"])
        config = JudgeConfig(**{**self.config.model_dump(), "metrics": ["tool_correctness", "faithfulness", "answer_correctness"], "thresholds": {}})
        with self.store.worker_lock():
            report = SemanticScorer(self.store, config, allow_api=True).score(attempt)
        self.assertEqual(report["status"], "completed", report)
        metrics = {m["metric_id"]: m for m in report["metrics"]}
        self.assertEqual(metrics["tool_correctness"]["value"], 0)
        self.assertEqual(metrics["faithfulness"]["status"], "missing_data")
        self.assertEqual(metrics["answer_correctness"]["status"], "missing_data")
        self.assertFalse(self.requests)

    def test_live_judge_and_cancellation(self):
        with patch.dict(os.environ, {"TEST_JUDGE_API_KEY": "fixture-local-key"}):
            attempt = self.attempt(judge_config=self.config.model_dump())
        scores = self.store.scores(attempt["attempt_id"])
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["status"], "completed")
        self.assertEqual(self.store.get_run(attempt["run_id"])["status"], "completed")
        with self.store.worker_lock():
            report = SemanticScorer(self.store, self.config, allow_api=True).score(attempt, cancelled=lambda: True)
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(len(self.requests), 2)


if __name__ == "__main__":
    unittest.main()
