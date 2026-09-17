import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_local_data import fixture_data

from evals.run_benchmark import build_evaluators, run_benchmark
from src.models import CineResult, Recommendation


def recommendation(fact, *, primary=True, **changes):
    values = {
        "watchlist_id": fact.watchlist_id,
        "title": fact.title,
        "year": fact.year,
        "fit_reason": "Pode combinar com o pedido atual.",
        "overview": fact.overview,
        "genres": fact.genres,
        "directors": fact.directors,
        "runtime_minutes": fact.runtime_minutes,
        "cast": fact.cast,
        "is_primary": primary,
    }
    values.update(changes)
    return Recommendation(**values)


def evaluate(evaluator, request, result):
    run = SimpleNamespace(outputs=result.model_dump(mode="json"))
    example = SimpleNamespace(inputs={"request": request})
    return evaluator(run, example)["score"]


class ExperimentTrackingTests(unittest.TestCase):
    def setUp(self):
        self.data, self.cache = fixture_data()
        self.recommender = Mock(data=self.data, cache=self.cache)

    def test_runner_tracks_real_recommender_requests_through_evaluate(self):
        recommender = Mock()
        recommender.recommend.side_effect = ["first-result", "second-result"]
        evaluate = Mock(return_value="experiment-result")
        result = run_benchmark(
            recommender,
            ["pedido real 1", "pedido real 2"],
            evaluate_fn=evaluate,
            experiment_prefix="letterboxd-recommender",
        )
        self.assertEqual("experiment-result", result)
        target = evaluate.call_args.args[0]
        self.assertEqual("first-result", target({"request": "pedido real 1"}))
        self.assertEqual("second-result", target({"request": "pedido real 2"}))
        self.assertEqual(
            [
                unittest.mock.call("pedido real 1"),
                unittest.mock.call("pedido real 2"),
            ],
            recommender.recommend.call_args_list,
        )
        self.assertEqual(
            [
                {"inputs": {"request": "pedido real 1"}},
                {"inputs": {"request": "pedido real 2"}},
            ],
            evaluate.call_args.kwargs["data"],
        )
        self.assertEqual(
            "letterboxd-recommender",
            evaluate.call_args.kwargs["experiment_prefix"],
        )
        evaluators = evaluate.call_args.kwargs["evaluators"]
        self.assertEqual(
            ["eligible_watchlist", "request_constraints", "verified_output"],
            [evaluator.__name__ for evaluator in evaluators],
        )

    def test_runner_reuses_existing_dataset_without_recreating_examples(self):
        client = Mock()
        client.has_dataset.return_value = True
        evaluate_fn = Mock(return_value="experiment-result")

        with (
            patch("evals.run_benchmark.Client", return_value=client),
            patch("evals.run_benchmark.evaluate", evaluate_fn),
        ):
            result = run_benchmark(
                Mock(),
                ["pedido real"],
                evaluate_fn=evaluate_fn,
                experiment_prefix="letterboxd-recommender",
            )

        self.assertEqual("experiment-result", result)
        client.create_dataset.assert_not_called()
        client.create_examples.assert_not_called()
        self.assertEqual(
            "letterboxd-recommender-requests",
            evaluate_fn.call_args.kwargs["data"],
        )

    def test_runner_creates_missing_dataset_and_examples(self):
        client = Mock()
        client.has_dataset.return_value = False
        evaluate_fn = Mock(return_value="experiment-result")

        with (
            patch("evals.run_benchmark.Client", return_value=client),
            patch("evals.run_benchmark.evaluate", evaluate_fn),
        ):
            result = run_benchmark(
                Mock(),
                ["pedido real 1", "pedido real 2"],
                evaluate_fn=evaluate_fn,
                experiment_prefix="letterboxd-recommender",
            )

        self.assertEqual("experiment-result", result)
        client.create_dataset.assert_called_once()
        client.create_examples.assert_called_once()
        self.assertEqual(
            [
                {"request": "pedido real 1"},
                {"request": "pedido real 2"},
            ],
            [
                example["inputs"]
                for example in client.create_examples.call_args.kwargs["examples"]
            ],
        )
        self.assertEqual(
            "letterboxd-recommender-requests",
            evaluate_fn.call_args.kwargs["data"],
        )

    def test_eligible_watchlist_accepts_intersection_and_rejects_foreign_movie(self):
        eligible_watchlist, _, _ = build_evaluators(self.recommender)
        fact = self.cache.movies["film:0"]
        valid = CineResult(recommendations=(recommendation(fact),))
        foreign = valid.model_copy(
            update={
                "recommendations": (
                    recommendation(fact, watchlist_id="outside"),
                )
            }
        )
        self.assertEqual(evaluate(eligible_watchlist, "qualquer", valid), 1)
        self.assertEqual(evaluate(eligible_watchlist, "qualquer", foreign), 0)

    def test_request_constraints_rejects_runtime_violation(self):
        _, request_constraints_evaluator, _ = build_evaluators(self.recommender)
        fact = self.cache.movies["film:0"]
        valid = CineResult(
            recommendations=(recommendation(fact, runtime_minutes=119),)
        )
        invalid = CineResult(
            recommendations=(recommendation(fact, runtime_minutes=120),)
        )
        request = "Quero um filme com menos de duas horas"
        self.assertEqual(evaluate(request_constraints_evaluator, request, valid), 1)
        self.assertEqual(evaluate(request_constraints_evaluator, request, invalid), 0)

    def test_verified_output_checks_facts_and_single_valid_primary(self):
        _, _, verified_output = build_evaluators(self.recommender)
        fact = self.cache.movies["film:0"]
        valid = CineResult(recommendations=(recommendation(fact),))
        divergent = CineResult(
            recommendations=(recommendation(fact, title="Título divergente"),)
        )
        invalid_primary = CineResult(
            recommendations=(recommendation(fact, primary=False),)
        )
        self.assertEqual(evaluate(verified_output, "qualquer", valid), 1)
        self.assertEqual(evaluate(verified_output, "qualquer", divergent), 0)
        self.assertEqual(evaluate(verified_output, "qualquer", invalid_primary), 0)
