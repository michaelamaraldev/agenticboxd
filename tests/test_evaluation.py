import unittest
from types import SimpleNamespace

from test_local_data import fixture_data

from evals.evaluation import (
    build_evaluation_cases,
    evaluate_data,
    evaluator_efficiency,
    evaluator_evidence_grounded,
    evaluator_quality,
    evaluator_routing,
    evaluator_structured_output,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.data, self.cache = fixture_data()
        self.recommender = SimpleNamespace(data=self.data, cache=self.cache)

    def test_data_evaluation_reports_duplicates_missing_cache_and_coverage(self):
        report = evaluate_data(self.data, self.cache)
        self.assertEqual(report.watchlist_count, 25)
        self.assertEqual(report.eligible_count, 24)
        self.assertEqual(report.missing_cache_count, 1)
        self.assertEqual(report.duplicate_watchlist_ids, 0)
        self.assertEqual(report.invalid_watchlist_rows, 0)
        self.assertAlmostEqual(report.cache_coverage, 24 / 25)
        self.assertTrue(report.valid)

    def test_data_evaluation_rejects_duplicate_or_invalid_watchlist_rows(self):
        first = self.data.watchlist[0]
        invalid = first.model_copy(update={"watchlist_id": "", "year": 1800})
        data = self.data.__class__(
            watchlist=(first, first, invalid),
            ratings_tsv=self.data.ratings_tsv,
            reviews_tsv=self.data.reviews_tsv,
            liked_films_tsv=self.data.liked_films_tsv,
            favorite_films_tsv=self.data.favorite_films_tsv,
            rankings_tsv=self.data.rankings_tsv,
            diary_tsv=self.data.diary_tsv,
        )
        report = evaluate_data(data, self.cache)
        self.assertEqual(report.duplicate_watchlist_ids, 1)
        self.assertEqual(report.invalid_watchlist_rows, 1)
        self.assertFalse(report.valid)

    def test_default_dataset_has_fifteen_cases_and_expected_dimensions(self):
        cases = build_evaluation_cases(self.recommender)
        self.assertEqual(len(cases), 15)
        self.assertTrue(any(case.requires_history for case in cases))
        self.assertTrue(any(case.expected_result_type == "empty" for case in cases))
        self.assertTrue(all("runtime_max" in case.required_constraints for case in cases))

    def test_structured_output_requires_one_primary_when_nonempty(self):
        valid = SimpleNamespace(
            outputs={"recommendations": [{"is_primary": True}]}
        )
        invalid = SimpleNamespace(
            outputs={"recommendations": [{"is_primary": True}, {"is_primary": True}]}
        )
        self.assertEqual(evaluator_structured_output(valid, None)["score"], 1)
        self.assertEqual(evaluator_structured_output(invalid, None)["score"], 0)

    def test_model_evaluators_check_evidence_efficiency_and_routing(self):
        run = SimpleNamespace(
            outputs={
                "recommendations": [{"is_primary": True, "evidence": {"request": "pedido"}}],
                "flow": ["recommendation agent chamou a tool analyze_history"],
                "metrics": {"model_calls": 6, "tool_calls": 6},
            }
        )
        example = SimpleNamespace(inputs={"request": "pedido", "requires_history": True})
        self.assertEqual(evaluator_evidence_grounded(run, example)["score"], 1)
        self.assertEqual(evaluator_efficiency(run, example)["score"], 1)
        self.assertEqual(evaluator_routing(run, example)["score"], 1)

    def test_quality_accepts_expected_watchlist_and_empty_cases(self):
        run = SimpleNamespace(
            outputs={
                "recommendations": [{"is_primary": True, "watchlist_id": "film:1"}]
            }
        )
        recommendation_case = SimpleNamespace(
            inputs={
                "acceptable_watchlist_ids": ["film:1"],
                "expected_result_type": "recommendation",
            }
        )
        empty_case = SimpleNamespace(
            inputs={"acceptable_watchlist_ids": [], "expected_result_type": "empty"}
        )
        self.assertEqual(evaluator_quality(run, recommendation_case)["score"], 2)
        self.assertEqual(evaluator_quality(SimpleNamespace(outputs={"recommendations": []}), empty_case)["score"], 1)


if __name__ == "__main__":
    unittest.main()
