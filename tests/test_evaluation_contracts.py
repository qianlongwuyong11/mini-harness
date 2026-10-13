import json
import sys
import unittest
from pydantic import ValidationError
from mini_harness.evaluation.models import (EvaluationDataset, MetricResult, TraceIdentity, ArtifactReference, RunRecord)
from mini_harness.evaluation.registry import validate_metric_result, metric_catalog
from mini_harness.evaluation.backend import worker_environment, require_judge

class EvaluationContractTests(unittest.TestCase):
    def dataset(self):
        return dict(dataset_id="fixture",cases=[dict(case_id="empty",category="scope",split="regression",scenario="empty scope",turns=[dict(input="question",selected_papers=[])],expected_behavior="ask to select")])

    def test_roundtrip_and_empty_scope(self):
        obj=EvaluationDataset.model_validate(self.dataset())
        self.assertEqual(EvaluationDataset.model_validate_json(obj.model_dump_json()),obj)
        self.assertEqual(obj.cases[0].turns[0].selected_papers,[])

    def test_rejects_ambiguous_or_executable_dataset(self):
        for mutate in (
            lambda x:x.update(schema_version=2),
            lambda x:x.update(python="run code"),
            lambda x:x['cases'].append(x['cases'][0].copy()),
            lambda x:x['cases'][0]['turns'][0].update(selected_papers=['unknown']),
            lambda x:x['cases'][0].update(budgets={'timeout_seconds':True}),
        ):
            data=self.dataset();mutate(data)
            with self.assertRaises(ValidationError):EvaluationDataset.model_validate(data)

    def test_missing_scores_are_not_zero_or_pass(self):
        base=dict(metric_id="recall_at_5",scope="rag",target_id="q1",unit="ratio",status="missing_data",reason="no gold")
        validate_metric_result(MetricResult(**base))
        for extra in (dict(value=0.0),dict(passed=True),dict(reason=None)):
            with self.assertRaises(ValidationError):MetricResult(**(base|extra))
        with self.assertRaises(ValueError):
            validate_metric_result(MetricResult(**(base|dict(status="scored",value=2.0))))

    def test_identity_and_artifact_traversal(self):
        with self.assertRaises(ValidationError):TraceIdentity(run_id="r",case_id="c",attempt_id="a",step_event_id=1)
        for path in ('../x','C:/x','/x','a/../../x','a\\..\\x'):
            with self.assertRaises(ValidationError):ArtifactReference(relative_path=path,sha256='a'*64,size_bytes=1)

    def test_run_completion_does_not_imply_success(self):
        run=RunRecord(run_id="r",dataset_sha256='a'*64,configuration_sha256='b'*64,mode="live_rules",status="completed")
        self.assertEqual(run.verdict,"incomplete")
        with self.assertRaises(ValidationError):RunRecord(**(run.model_dump()|dict(mode="live_judge")))

    def test_child_environment_does_not_mutate_parent(self):
        source={'CONFIDENT_API_KEY':'secret','DEEPEVAL_TELEMETRY_OPT_OUT':'0','PATH':'keep'}
        child=worker_environment(source)
        self.assertEqual(source['CONFIDENT_API_KEY'],'secret')
        self.assertNotIn('CONFIDENT_API_KEY',child)
        self.assertEqual(child['DEEPEVAL_DISABLE_LEGACY_KEYFILE'],'1')
        self.assertEqual(child['PATH'],'keep')
        with self.assertRaises(ValueError):require_judge(None)
        with self.assertRaises(ValueError):require_judge('default')

    def test_catalog_serializable_and_semantics(self):
        rows=metric_catalog();json.dumps(rows)
        self.assertEqual(len(rows),len({r['metric_id'] for r in rows}))
        self.assertEqual(next(r for r in rows if r['metric_id']=='rrf_score')['unit'],'score')
        self.assertNotIn('deepeval',sys.modules)

if __name__=='__main__':unittest.main()
