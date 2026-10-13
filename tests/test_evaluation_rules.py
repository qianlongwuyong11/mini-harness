import tempfile
from pathlib import Path
import unittest
from mini_harness.evaluation.models import EvaluationDataset
from mini_harness.evaluation.rules import score_case
from mini_harness.evaluation.metrics import retrieval_metrics, read_artifact
from mini_harness.evaluation.trace import TraceStore
from mini_harness.session import Session


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.artifacts = TraceStore(self.tmp.name)
        self.dataset = EvaluationDataset.model_validate(dict(dataset_id="d", papers=[dict(logical_key="p", paper_id="p", version="v", content_sha256="a"*64)], cases=[dict(case_id="c", category="scope", split="regression", scenario="s", turns=[dict(input="q",selected_papers=["p"])], expected_behavior="good", gold_evidence=[dict(paper_key="p",content_sha256="a"*64,pages=[1],chunk_ids=["c1","c2"])])]))

    def score(self, entries):
        session = Session()
        session.append("turn/start",{})
        for kind, data in entries:
            session.append(kind,data)
        session.append("turn/end",{"reason":"completed"})
        rows = score_case(self.dataset.cases[0],self.dataset,session.events,self.artifacts.root)
        return {r.metric_id:r for r in rows}

    def test_llamaindex_matches_known_historical_formula(self):
        rows = {m.metric_id:m for m in retrieval_metrics("q",["x","c2","c1"],["c1","c2"],top_k=5)}
        self.assertEqual(rows["recall_at_1"].value,0)
        self.assertEqual(rows["hit_at_3"].value,1)
        self.assertEqual(rows["recall_at_5"].value,1)
        self.assertEqual(rows["mrr_at_5"].value,.5)
        self.assertEqual(retrieval_metrics("q",[],["c1"],top_k=5)[0].value,0)
        self.assertEqual(retrieval_metrics("q",[],[],top_k=5)[0].status,"missing_data")
        self.assertEqual(retrieval_metrics("q",["c1"],["c1"],top_k=1)[-1].status,"missing_data")

    def test_scope_and_forged_citation_fail(self):
        rows = self.score([("tool/call",dict(name="read_section",call_id="x",arguments='{"paper_id":"other"}')),
                           ("assistant/message",dict(content="claim [cite:forged]",tool_calls=[]))])
        self.assertEqual(rows["scope_violations"].value,1)
        self.assertFalse(rows["scope_violations"].passed)
        self.assertEqual(rows["invalid_citations"].value,1)
        self.assertEqual(rows["citation_validity"].value,0)

    def test_evidence_retrieval_injection_and_citation(self):
        flow = self.artifacts.put(dict(returned=[dict(paper_id="p",version="v",chunk_id="c1")],decisions=[dict(reason="accepted",evidence_id="e",paper_id="p",version="v",chunk_id="c1")]))
        retrieval = self.artifacts.put(dict(returned_chunk_ids=["c1"],top_k=5))
        rows=self.score([("rag/evidence-flow",dict(detail_ref=flow)),("rag/retrieval",dict(detail_ref=retrieval,status="ok")),
                         ("rag/context",dict(evidence_ids=["e"])),("rag/citations",dict(ids=["e"])),
                         ("assistant/message",dict(content="claim [cite:e]",tool_calls=[]))])
        self.assertEqual(rows["case_gold_coverage"].value,.5)
        self.assertEqual(rows["injected_gold_coverage"].value,.5)
        self.assertEqual(rows["recall_at_5"].value,.5)
        self.assertEqual(rows["invalid_citations"].value,0)

    def test_integrity_and_missing_trace_not_zero(self):
        ref=self.artifacts.put({"a":1})
        (self.artifacts.root/ref["relative_path"]).write_text('changed')
        with self.assertRaises(ValueError):read_artifact(self.artifacts.root,ref)
        rows=self.score([("rag/evidence-flow",dict(detail_ref=ref))])
        self.assertEqual(rows["case_gold_coverage"].status,"missing_data")
        self.assertIsNone(rows["case_gold_coverage"].value)

    def test_tool_recovery_does_not_imply_semantic_success(self):
        rows=self.score([("metrics/tool",dict(name="list_papers",status="error")),("metrics/tool",dict(name="list_papers",status="ok"))])
        self.assertEqual(rows["tool_recoveries"].value,1)
        self.assertEqual(rows["recovery_success"].status,"missing_data")

if __name__ == "__main__":unittest.main()
