from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.local_data import request_constraints
from src.models import LetterboxdData, TmdbCache

VERIFIED_FIELDS = (
    "watchlist_id",
    "title",
    "year",
    "overview",
    "genres",
    "directors",
    "runtime_minutes",
    "cast",
)


@dataclass(frozen=True, slots=True)
class DataEvaluation:
    watchlist_count: int
    eligible_count: int
    missing_cache_count: int
    duplicate_watchlist_ids: int
    invalid_watchlist_rows: int
    missing_runtime_count: int
    missing_required_fields: int
    cache_coverage: float
    valid: bool

    def model_dump(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    request: str
    acceptable_candidate_ids: tuple[int, ...]
    acceptable_watchlist_ids: tuple[str, ...]
    required_constraints: dict[str, int | None]
    requires_history: bool
    expected_result_type: str
    manual_quality_score: int | None = None

    def inputs(self) -> dict[str, object]:
        return {
            "request": self.request,
            "acceptable_candidate_ids": list(self.acceptable_candidate_ids),
            "acceptable_watchlist_ids": list(self.acceptable_watchlist_ids),
            "required_constraints": self.required_constraints,
            "requires_history": self.requires_history,
            "expected_result_type": self.expected_result_type,
            "manual_quality_score": self.manual_quality_score,
        }


def evaluate_data(data: LetterboxdData, cache: TmdbCache) -> DataEvaluation:
    watchlist_ids = [film.watchlist_id for film in data.watchlist]
    duplicate_count = len(watchlist_ids) - len(set(watchlist_ids))
    invalid_rows = sum(
        not film.watchlist_id.strip() or not film.title.strip() or film.year < 1888
        for film in data.watchlist
    )
    eligible_ids = set(watchlist_ids).intersection(cache.movies)
    missing_cache_count = len(set(watchlist_ids) - set(cache.movies))
    required_fields = ("title", "year")
    missing_required_fields = sum(
        any(
            getattr(fact, field) is None
            or getattr(fact, field) == ""
            or getattr(fact, field) == ()
            for field in required_fields
        )
        for fact in cache.movies.values()
        if fact.watchlist_id in eligible_ids
    )
    missing_runtime_count = sum(
        fact.runtime_minutes is None
        for fact in cache.movies.values()
        if fact.watchlist_id in eligible_ids
    )
    coverage = len(eligible_ids) / len(set(watchlist_ids)) if watchlist_ids else 0.0
    valid = not duplicate_count and not invalid_rows and not missing_required_fields
    return DataEvaluation(
        watchlist_count=len(watchlist_ids),
        eligible_count=len(eligible_ids),
        missing_cache_count=missing_cache_count,
        duplicate_watchlist_ids=duplicate_count,
        invalid_watchlist_rows=invalid_rows,
        missing_runtime_count=missing_runtime_count,
        missing_required_fields=missing_required_fields,
        cache_coverage=coverage,
        valid=valid,
    )


def build_evaluation_cases(recommender: Any) -> tuple[EvaluationCase, ...]:
    from src.local_data import LocalCatalog

    catalog = LocalCatalog(recommender.data, recommender.cache)
    records = catalog.records
    ids = {
        int(record["candidate_id"]): str(record["watchlist_id"]) for record in records
    }
    short = [
        int(r["candidate_id"])
        for r in records
        if isinstance(r["runtime_minutes"], int) and r["runtime_minutes"] <= 90
    ]
    recent = [
        int(r["candidate_id"])
        for r in records
        if isinstance(r["year"], int) and r["year"] >= 2000
    ]
    drama = [
        int(r["candidate_id"])
        for r in records
        if "drama" in {str(g).casefold() for g in r["genres"]}
    ]
    old = [
        int(r["candidate_id"])
        for r in records
        if isinstance(r["year"], int) and r["year"] < 1960
    ]
    cases = (
        ("Quero um drama para assistir hoje", drama[:5], False, "recommendation"),
        ("Escolha um filme de até 90 minutos", short[:5], False, "recommendation"),
        ("Quero algo lançado depois de 2000.", recent[:5], False, "recommendation"),
        (
            "Quero um filme antigo que não seja dos estados unidos",
            old[:5],
            False,
            "recommendation",
        ),
        (
            "Quero algo diferente do que assisti recentemente",
            drama[:5],
            True,
            "recommendation",
        ),
        (
            "recomende algo parecido com After Hours do Scorsese",
            drama[:5],
            False,
            "recommendation",
        ),
        ("O que combina com meu gosto por dramas?", drama[:5], False, "recommendation"),
        (
            "Quero um filme curto e lançado depois de 2000",
            [i for i in short if i in recent][:5],
            False,
            "recommendation",
        ),
        (
            "Escolha um suspense com menos de duas horas",
            [
                int(r["candidate_id"])
                for r in records
                if "thriller" in {str(g).casefold() for g in r["genres"]}
                and isinstance(r["runtime_minutes"], int)
                and r["runtime_minutes"] < 120
            ][:5],
            False,
            "recommendation",
        ),
        ("Quero ver algum filme japonês", drama[:5], False, "recommendation"),
        (
            "Recomende qualquer filme da minha watchlist, mas me de um motivo claro",
            list(ids)[:5],
            False,
            "recommendation",
        ),
        ("Quero uma ficção científica lançada antes de 2000", [], False, "empty"),
        ("Quero algo pra rir bastante hoje", [], False, "empty"),
        ("Escolhe um filme levinho", list(ids)[:5], False, "recommendation"),
        (
            "Quero um filme de até duas horas",
            [i for i in short if i in ids][:5],
            False,
            "recommendation",
        ),
    )
    normalized = []
    for item in cases:
        if len(item) == 1:
            request = item[0]
            candidate_ids, requires_history, result_type = (
                list(ids)[:5],
                False,
                "recommendation",
            )
        else:
            request, candidate_ids, requires_history, result_type = item
        candidate_ids = tuple(candidate_ids)
        watchlist_ids = tuple(
            ids[candidate_id] for candidate_id in candidate_ids if candidate_id in ids
        )
        constraints = request_constraints(request)
        normalized.append(
            EvaluationCase(
                request=request,
                acceptable_candidate_ids=candidate_ids,
                acceptable_watchlist_ids=watchlist_ids,
                required_constraints={
                    "year_min": constraints.year_min,
                    "year_max": constraints.year_max,
                    "runtime_min": constraints.runtime_min,
                    "runtime_max": constraints.runtime_max,
                },
                requires_history=requires_history,
                expected_result_type=result_type,
            )
        )
    return tuple(normalized)


def cases_as_dataset(cases: Sequence[EvaluationCase]) -> list[dict[str, object]]:
    return [{"inputs": case.inputs()} for case in cases]


def _outputs(run: Any) -> Mapping[str, Any] | None:
    outputs = getattr(run, "outputs", None)
    return outputs if isinstance(outputs, Mapping) else None


def _recommendations(run: Any) -> list[Mapping[str, Any]] | None:
    outputs = _outputs(run)
    recommendations = outputs.get("recommendations") if outputs else None
    if not isinstance(recommendations, list) or not all(
        isinstance(item, Mapping) for item in recommendations
    ):
        return None
    return recommendations


def evaluator_structured_output(run: Any, example: Any) -> dict[str, int]:
    recommendations = _recommendations(run)
    if recommendations is None:
        return {"score": 0}
    primary_count = sum(item.get("is_primary") is True for item in recommendations)
    return {"score": int(not recommendations or primary_count == 1)}


def evaluator_evidence_grounded(run: Any, example: Any) -> dict[str, int]:
    recommendations = _recommendations(run)
    if recommendations is None:
        return {"score": 0}
    valid = all(
        isinstance(item.get("evidence"), Mapping) and bool(item["evidence"])
        for item in recommendations
    )
    return {"score": int(valid)}


def evaluator_efficiency(run: Any, example: Any) -> dict[str, int]:
    outputs = _outputs(run)
    metrics = outputs.get("metrics", {}) if outputs else {}
    if not isinstance(metrics, Mapping):
        return {"score": 0}
    model_calls = metrics.get("model_calls", 0)
    tool_calls = metrics.get("tool_calls", 0)
    return {
        "score": int(
            isinstance(model_calls, int)
            and isinstance(tool_calls, int)
            and model_calls <= 6
            and tool_calls <= 6
        )
    }


def evaluator_routing(run: Any, example: Any) -> dict[str, int]:
    outputs = _outputs(run)
    flow = outputs.get("flow", []) if outputs else []
    request = str(example.inputs.get("request", ""))
    requires_history = bool(example.inputs.get("requires_history", False))
    saw_history = (
        any("analyze_history" in str(event) for event in flow)
        if isinstance(flow, list)
        else False
    )
    if not requires_history and saw_history:
        return {"score": 0}
    if requires_history and not saw_history:
        return {"score": 0}
    return {"score": int(bool(request.strip()))}


def evaluator_quality(run: Any, example: Any) -> dict[str, int | float]:
    recommendations = _recommendations(run)
    acceptable = set(example.inputs.get("acceptable_watchlist_ids", []))
    expected_result_type = example.inputs.get("expected_result_type", "recommendation")
    if recommendations is None:
        return {"score": 0}
    if expected_result_type == "empty":
        return {"score": int(not recommendations)}
    if not recommendations:
        return {"score": 0}
    primary = next(
        (item for item in recommendations if item.get("is_primary") is True),
        recommendations[0],
    )
    if primary.get("watchlist_id") in acceptable:
        return {"score": 2}
    return {"score": 1}
