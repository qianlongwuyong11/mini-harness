import json
from pathlib import Path
import tempfile
import unittest

from mini_harness.client import Message, ToolCall
from mini_harness.evaluation.dataset import load_dataset, bind_papers
from mini_harness.evaluation.models import EvaluationDataset
from mini_harness.evaluation.store import EvaluationStore
from mini_harness.evaluation.runner import EvaluationWorker


def dataset(turns=1, cases=1, **budgets):
    return EvaluationDataset.model_validate(dict(dataset_id="fixture", cases=[dict(
        case_id=f"case{i}", category="smoke", split="regression", scenario="private-scenario",
        expected_behavior="SECRET-GOLD", reference_answer="SECRET-ANSWER", key_points=["SECRET-POINT"],
        turns=[dict(input=f"question-{i}-{j}", selected_papers=[]) for j in range(turns)],
        budgets=budgets) for i in range(cases)]))


class Client:
    MODEL = "fake-local"
    def __init__(self):
        self.messages = []
    def chat(self, messages, tools):
        self.messages.append(messages)
        return Message(role="assistant", content="answer", usage={"total_tokens": 5})


class EvaluationQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = EvaluationStore(self.root / "evals")

    def test_json_jsonl_validation_and_path_scope(self):
        value = dataset().model_dump()
        (self.root / "data.json").write_text(json.dumps(value), encoding="utf-8")
        self.assertEqual(load_dataset(self.root, "data.json"), dataset())
        cases = value.pop("cases")
        (self.root / "data.jsonl").write_text("\n".join(json.dumps(x) for x in [value, *cases]), encoding="utf-8")
        self.assertEqual(load_dataset(self.root, "data.jsonl"), dataset())
        for path in ("../data.json", str(self.root / "data.json")):
            with self.assertRaises(ValueError): load_dataset(self.root, path)
        (self.root / "bad.json").write_text('{"dataset_id":"a","dataset_id":"b"}')
        with self.assertRaises(ValueError): load_dataset(self.root, "bad.json")

    def test_isolation_multiturn_and_no_gold_in_model(self):
        clients = []
        def factory():
            client = Client()
            clients.append(client)
            return client
        run = self.store.enqueue(dataset(turns=2, cases=2))
        EvaluationWorker(self.store, factory).work()
        self.assertEqual(self.store.get_run(run)["status"], "completed")
        self.assertEqual(self.store.get_run(run)["verdict"], "incomplete")
        attempts = self.store.attempts(run)
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0]["session_id"], attempts[1]["session_id"])
        for i, client in enumerate(clients):
            self.assertEqual(len(client.messages), 2)
            texts = repr(client.messages)
            self.assertNotIn("SECRET", texts)
            self.assertNotIn("private-scenario", texts)
            self.assertNotIn(f"question-{1-i}", texts)
            self.assertIn(f"question-{i}-0", repr(client.messages[1]))
        directory = self.store.root / attempts[0]["artifact_dir"]
        self.assertTrue((directory / "trace.jsonl").exists())
        events = [json.loads(line) for line in (directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()][1:]
        metrics = [e for e in events if e["type"] == "metrics/model"]
        self.assertEqual(metrics[0]["data"]["run_id"], run)

    def test_cancel_retains_partial_trace_retry_gets_new_attempt(self):
        run = self.store.enqueue(dataset(turns=2))
        owner = self
        class CancelClient(Client):
            def chat(self, messages, tools):
                owner.store.cancel(run)
                return super().chat(messages, tools)
        EvaluationWorker(self.store, CancelClient).work()
        self.assertEqual(self.store.get_run(run)["status"], "interrupted")
        old = self.store.attempts(run)[0]
        self.assertEqual(old["status"], "interrupted")
        self.store.retry(run)
        EvaluationWorker(self.store, Client).work()
        attempts = self.store.attempts(run)
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0]["attempt_id"], attempts[1]["attempt_id"])
        self.assertEqual(self.store.get_run(run)["status"], "completed")

    def test_worker_lock_and_restart_recovery(self):
        run = self.store.enqueue(dataset())
        with self.store.worker_lock():
            self.store.claim()
            self.store.start_attempt(run, "case0")
            with self.assertRaises(RuntimeError):
                EvaluationWorker(self.store, Client).work()
            self.assertEqual(self.store.get_run(run)["status"], "running")
        EvaluationWorker(self.store, Client).work()
        self.assertEqual(self.store.get_run(run)["status"], "interrupted")
        self.assertEqual(self.store.attempts(run)[0]["status"], "interrupted")

    def test_budgets_and_no_write_tools(self):
        class LoopClient(Client):
            def chat(inner, messages, tools):
                self.assertEqual(tools, [])
                return Message(role="assistant", tool_calls=(ToolCall("x", "shell", "{}"),), usage={"total_tokens": 5})
        run = self.store.enqueue(dataset(max_steps_per_turn=1))
        EvaluationWorker(self.store, LoopClient).work()
        self.assertEqual(self.store.get_run(run)["status"], "failed")
        run2 = self.store.enqueue(dataset(turns=2, max_total_tokens=5))
        c = Client()
        EvaluationWorker(self.store, lambda: c).work()
        self.assertEqual(len(c.messages), 1)
        self.assertEqual(self.store.get_run(run2)["status"], "failed")

    def test_mismatch_before_model_and_mode_rejection(self):
        from types import SimpleNamespace
        data = dataset()
        value = data.model_dump()
        value["papers"] = [dict(logical_key="paper", paper_id="p", version="v", content_sha256="a"*64)]
        data = EvaluationDataset.model_validate(value)
        catalog = SimpleNamespace(get_paper=lambda _: object(), get_active_version=lambda _: SimpleNamespace(version="other", source_hash="a"*64))
        with self.assertRaises(ValueError): bind_papers(data, catalog)
        with self.assertRaises(ValueError): self.store.enqueue(data, mode="live_judge")
        run = self.store.enqueue(data)
        c = Client()
        EvaluationWorker(self.store, lambda: c, SimpleNamespace(catalog=catalog)).work()
        self.assertEqual(c.messages, [])
        self.assertEqual(self.store.get_run(run)["status"], "failed")

    def test_real_rag_plugin_selection_and_artifacts(self):
        from dataclasses import replace
        from mini_harness.rag.runtime import PaperRuntime, RagConfig
        from mini_harness.rag.models import Paper, PaperVersion, VersionState
        runtime = PaperRuntime(RagConfig(catalog_path=str(self.root / "catalog.sqlite3")))
        self.addCleanup(runtime.close)
        catalog = runtime.catalog
        catalog.create_paper(Paper("p", "fixture://paper", "a"*64, "Fixture paper"))
        version = PaperVersion("p", "v", "a"*64, "parser", "chunker")
        catalog.save_version(version)
        with catalog._connection(write=True) as db:
            db.execute("UPDATE versions SET data=?", (replace(version, state=VersionState.READY, chunk_count=1).to_json(),))
        catalog.activate_version("p", "v")
        data = dataset().model_dump()
        data["papers"] = [dict(logical_key="paper", paper_id="p", version="v", content_sha256="a"*64)]
        data["cases"][0]["turns"][0]["selected_papers"] = ["paper"]
        owner = self
        class PaperClient(Client):
            def chat(inner, messages, tools):
                owner.assertEqual({t.name for t in tools}, {"list_papers", "search_papers", "read_section"})
                if not inner.messages:
                    inner.messages.append(messages)
                    return Message(role="assistant", tool_calls=(ToolCall("list", "list_papers", "{}"),), usage={"total_tokens": 5})
                body = next(m.content for m in messages if m.role == "tool")
                owner.assertIn("Fixture paper", body)
                owner.assertNotIn("SECRET", repr(messages))
                return Message(role="assistant", content="本轮检索证据未提供该信息", usage={"total_tokens": 5})
        run = self.store.enqueue(EvaluationDataset.model_validate(data))
        EvaluationWorker(self.store, PaperClient, runtime).work()
        self.assertEqual(self.store.get_run(run)["status"], "completed")
        self.assertFalse(runtime._closed)

    def test_report_failure_does_not_leave_running_attempt(self):
        from unittest.mock import patch
        run = self.store.enqueue(dataset())
        # Trace writes also fail, but the successful provider call must not be repeated.
        c = Client()
        with patch("mini_harness.evaluation.runner.TraceStore.put", side_effect=OSError("disk full")):
            EvaluationWorker(self.store, lambda: c).work()
        self.assertEqual(len(c.messages), 1)
        self.assertEqual(self.store.get_run(run)["status"], "failed")
        self.assertEqual(self.store.attempts(run)[0]["status"], "failed")

    def test_cancel_queued_and_retry_keeps_completed_cases(self):
        run = self.store.enqueue(dataset(cases=2))
        self.store.cancel(run)
        client = Client()
        EvaluationWorker(self.store, lambda: client).work()
        self.assertFalse(client.messages)
        self.store.retry(run)
        count = 0
        def factory():
            nonlocal count
            count += 1
            if count == 2:
                raise ValueError("fixture failure")
            return Client()
        EvaluationWorker(self.store, factory).work()
        self.assertEqual([a["status"] for a in self.store.attempts(run)], ["completed", "failed"])
        self.store.retry(run)
        EvaluationWorker(self.store, Client).work()
        self.assertEqual([a["case_id"] for a in self.store.attempts(run)], ["case0", "case1", "case1"])
        self.assertEqual(self.store.get_run(run)["status"], "completed")


if __name__ == "__main__":
    unittest.main()
