"""Regression experiment accounting: no remote API or paper corpus required."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('ab_experiment', Path(__file__).resolve().parents[1] / 'scripts/evaluate_list_papers_ab.py')
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def row(case='a', status='completed', failures=()):
    return dict(run_id='run', case_id=case, variant='before', status=status, rule_failures=list(failures))


def review(case='a', verdict='passed'):
    return dict(run_id='run', case_id=case, verdict=verdict, reviewer='test-reviewer', reason='explicit rubric result')


class ExperimentAccountingTests(unittest.TestCase):
    def test_completed_without_review_is_not_pass(self):
        result = ab.summarize([row()], [])['summary']['before']
        self.assertEqual(result['pending_review'], 1)
        self.assertIsNone(result['pass_rate'])

    def test_failed_execution_stays_in_denominator(self):
        result = ab.summarize([row(), row('b', 'failed')], [review()])['summary']['before']
        self.assertEqual((result['passed'], result['failed'], result['total']), (1, 1, 2))
        self.assertEqual(result['pass_rate'], .5)

    def test_semantic_review_cannot_override_rule_failure(self):
        result = ab.summarize([row(failures=['scope_violations'])], [review()])
        self.assertEqual(result['cases'][0]['verdict'], 'failed')

    def test_unexecuted_is_neither_quality_failure_nor_zero_rate(self):
        result = ab.summarize([row(status='not_executed')], [])['summary']['before']
        self.assertEqual(result['not_executed'], 1)
        self.assertEqual(result['failed'], 0)
        self.assertIsNone(result['pass_rate'])

    def test_duplicate_unknown_and_unattributed_reviews_rejected(self):
        for reviews in ([review(), review()], [review('other')], [{**review(), 'reviewer':''}]):
            with self.assertRaises(ValueError): ab.summarize([row()], reviews)


if __name__ == '__main__': unittest.main()
