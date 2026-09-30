from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from langsmith import Client, evaluate

from app import Recommends, create_recommender
from evals.evaluation import (
    VERIFIED_FIELDS,
    build_evaluation_cases,
    cases_as_dataset,
    evaluate_data,
    evaluator_efficiency,
    evaluator_evidence_grounded,
    evaluator_quality,
    evaluator_routing,
    evaluator_structured_output,
)
from src.local_data import request_constraints as extract_request_constraints
from src.observability import RecommendationObserver

DEFAULT_REQUESTS = (
    "Quero um filme de até duas horas para assistir hoje.",
    "Quero algo diferente do que tenho assistido recentemente.",
    "Escolha um drama da minha watchlist lançado depois de 2000.",
)

def _recommendations(run: Any) -> list[Mapping[str, Any]] | None:
    outputs = getattr(run, "outputs", None)
    if not isinstance(outputs, Mapping):
        return None
    recommendations = outputs.get("recommendations", [])
    if not isinstance(recommendations, list):
        return None
    if not all(isinstance(item, Mapping) for item in recommendations):
        return None
    return recommendations


def build_evaluators(recommender: Recommends) -> tuple[Callable[..., Any], ...]:
    # Os avaliadores conferem regras objetivas sem pedir outra avaliação ao modelo
    data = getattr(recommender, "data", None)
    watchlist = getattr(data, "watchlist", ())
    if not isinstance(watchlist, (list, tuple)):
        watchlist = ()
    watchlist_ids = {film.watchlist_id for film in watchlist}
    recommender_cache = getattr(recommender, "cache", None)
    cache = getattr(recommender_cache, "movies", {})
    if not isinstance(cache, Mapping):
        cache = {}
    eligible_ids = watchlist_ids.intersection(cache)

    def eligible_watchlist(run: Any, example: Any) -> dict[str, int]:
        recommendations = _recommendations(run)
        if recommendations is None:
            return {"score": 0}
        valid = all(
            item.get("watchlist_id") in eligible_ids for item in recommendations
        )
        return {"score": int(valid)}

    def request_constraints(run: Any, example: Any) -> dict[str, int]:
        request = example.inputs.get("request", "")
        constraints = extract_request_constraints(str(request))
        recommendations = _recommendations(run)
        if recommendations is None:
            return {"score": 0}
        valid = True
        for item in recommendations:
            year = item.get("year")
            runtime = item.get("runtime_minutes")
            if constraints.year_min is not None and (
                not isinstance(year, int) or year < constraints.year_min
            ):
                valid = False
            if constraints.year_max is not None and (
                not isinstance(year, int) or year > constraints.year_max
            ):
                valid = False
            if constraints.runtime_min is not None and (
                not isinstance(runtime, int) or runtime < constraints.runtime_min
            ):
                valid = False
            if constraints.runtime_max is not None and (
                not isinstance(runtime, int) or runtime > constraints.runtime_max
            ):
                valid = False
        return {"score": int(valid)}

    def verified_output(run: Any, example: Any) -> dict[str, int]:
        recommendations = _recommendations(run)
        if recommendations is None:
            return {"score": 0}
        if recommendations:
            primaries = sum(item.get("is_primary") is True for item in recommendations)
            if primaries != 1:
                return {"score": 0}
        for item in recommendations:
            watchlist_id = item.get("watchlist_id")
            fact = cache.get(watchlist_id)
            if watchlist_id not in eligible_ids or fact is None:
                return {"score": 0}
            expected = fact.model_dump(mode="json")
            if any(item.get(field) != expected.get(field) for field in VERIFIED_FIELDS):
                return {"score": 0}
        return {"score": 1}

    return (
        eligible_watchlist,
        request_constraints,
        verified_output,
    )


def build_all_evaluators(recommender: Recommends) -> tuple[Callable[..., Any], ...]:
    return (
        *build_evaluators(recommender),
        evaluator_structured_output,
        evaluator_evidence_grounded,
        evaluator_routing,
        evaluator_efficiency,
        evaluator_quality,
    )


def data_evaluation(recommender: Recommends) -> dict[str, object]:
    return evaluate_data(recommender.data, recommender.cache).model_dump()


def run_benchmark(
    recommender: Recommends,
    requests: Sequence[str],
    *,
    evaluate_fn: Callable[..., Any] = evaluate,
    experiment_prefix: str = "letterboxd-recommender",
) -> Any:
    """Run real recommendation requests as a reproducible LangSmith experiment."""

    def target(example: dict[Any, Any]) -> dict[Any, Any]:
        result = recommender.recommend(example["request"])
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return cast(dict[Any, Any], result)

    cases = build_evaluation_cases(recommender) if not requests else ()
    data = cases_as_dataset(cases) if cases else [{"inputs": {"request": request}} for request in requests]
    evaluators = build_all_evaluators(recommender) if cases else build_evaluators(recommender)
    if evaluate_fn is not evaluate:
        return evaluate_fn(
            target,
            data=data,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
        )
    client = Client()
    observer = getattr(recommender, "observer", None)
    metadata = (
        observer.experiment_metadata()
        if isinstance(observer, RecommendationObserver)
        else None
    )
    dataset_name = (
        f"{experiment_prefix}-evaluation-v1"
        if cases
        else f"{experiment_prefix}-requests"
    )
    if not client.has_dataset(dataset_name=dataset_name):
        dataset_examples = cases_as_dataset(cases) if cases else data
        client.create_dataset(
            dataset_name,
            description="Pedidos reais usados na demonstração de tracking.",
        )
        client.create_examples(
            dataset_name=dataset_name,
            examples=[
                {
                    "id": uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_name}:{item['inputs']['request']}"),
                    "inputs": item["inputs"],
                }
                for item in dataset_examples
            ],
        )
    return evaluate_fn(
        target,
        data=dataset_name,
        experiment_prefix=experiment_prefix,
        client=client,
        metadata=metadata,
        evaluators=evaluators,
    )


def main() -> None:
    recommender = create_recommender()
    report = data_evaluation(recommender)
    print(report)
    run_benchmark(recommender, ())


if __name__ == "__main__":
    main()
