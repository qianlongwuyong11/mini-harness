import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rag_acceptance import metrics,latency
class AcceptanceMetricsTests(unittest.TestCase):
    def test_multiple_gold_and_duplicate_hits(self):
        m=metrics([(['a','a','x','b'],['a','b']),(['x','c'],['c'])])
        self.assertEqual(m['recall@1'],.25)
        self.assertEqual(m['recall@3'],1.)
        self.assertEqual(m['mrr@3'],.75)
        self.assertEqual(m['hit@1'],.5)
    def test_no_hits_and_empty_gold(self):
        self.assertEqual(metrics([([],['a'])])['mrr@5'],0)
        with self.assertRaises(ValueError):metrics([(['a'],[])])
        with self.assertRaises(ValueError):metrics([])
    def test_nearest_rank_percentile(self):
        self.assertEqual(latency(list(range(1,21)))['p95_ms'],19)
