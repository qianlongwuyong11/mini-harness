import unittest
from mini_harness.evaluation.benchmark import build_datasets, summarize


class BenchmarkTests(unittest.TestCase):
    def test_drafts_bound_and_splits_separate(self):
        data=build_datasets(dict(logical_key='paper',paper_id='p',version='v',content_sha256='a'*64))
        self.assertEqual(len(data['regression'].cases),4)
        self.assertEqual(len(data['heldout'].cases),3)
        self.assertEqual(data['regression'].cases[-1].turns[-1].selected_papers,[])
        self.assertTrue(all(c.split=='heldout' for c in data['heldout'].cases))
        self.assertTrue(all(not c.gold_evidence for d in data.values() for c in d.cases))

    def test_missing_runs_never_pass(self):
        result=summarize(None,[],[])
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(len(result['blockers']),2)
        with self.assertRaises(ValueError): summarize(None,['x','x'],[])
