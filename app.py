from __future__ import annotations

import csv
import io
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.agents.recommendation import (
    RequestBudget,
    create_recommendation_agent,
    recommendation_message,
)
from src.local_data import (
    EvidenceStore,
    LocalCatalog,
    SearchConstraints,
    TasteCache,
    create_search_candidates_tool,
    request_constraints,
)
from src.models import (
    CineResult,
    Film,
    LetterboxdData,
    Recommendation,
    RecommendationChoices,
    TasteProfile,
    TmdbCache,
)
from src.tmdb import TmdbClient, load_tmdb_cache, sync_tmdb_cache
from src.tools.analyze_history import HistoryToolState, create_analyze_history_tool
from src.tools.analyze_taste import TasteToolState, create_analyze_taste_tool
from src.tools.get_tmdb_details import TmdbToolState, create_get_tmdb_details_tool


class InvalidAgentResult(TypeError, ValueError):
    pass


def _rows(path: Path, required_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_fields = set(reader.fieldnames or ())
        missing = set(required_fields) - actual_fields
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def _required(row: dict[str, str], field: str, path: Path) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(f"Missing {field!r} in {path}")
    return value


def _tsv(headers: tuple[str, ...], rows: Sequence[Sequence[object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, dialect="excel-tab", lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()


def _optional_rows(path: Path, required_fields: tuple[str, ...]) -> list[dict[str, str]]:
    return _rows(path, required_fields) if path.exists() else []


def _ranking_rows(path: Path) -> list[tuple[str, str, str, str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Position,")), None
    )
    if header_index is None:
        raise ValueError(f"Missing ranking table in {path}")
    metadata = list(csv.DictReader(lines[1:3]))
    list_name = metadata[0].get("Name", path.stem) if metadata else path.stem
    entries = csv.DictReader(lines[header_index:])
    return [
        (list_name, row["Position"], row["Name"], row["Year"])
        for row in entries
        if row.get("Position") and row.get("Name") and row.get("Year")
    ]


def load_letterboxd(data_dir: Path) -> LetterboxdData:
    watchlist_path = data_dir / "watchlist.csv"
    ratings_path = data_dir / "ratings.csv"
    reviews_path = data_dir / "reviews.csv"
    watched_path = data_dir / "watched.csv"
    liked_films_path = data_dir / "likes" / "films.csv"
    profile_path = data_dir / "profile.csv"
    diary_path = data_dir / "diary.csv"
    watchlist_rows = _rows(watchlist_path, ("Name", "Year", "Letterboxd URI"))
    ratings_rows = _rows(ratings_path, ("Name", "Year", "Rating"))
    reviews_rows = _rows(reviews_path, ("Name", "Year", "Rating", "Review", "Tags"))
    watched_rows = _optional_rows(watched_path, ("Name", "Year", "Letterboxd URI"))
    liked_films_rows = _optional_rows(liked_films_path, ("Name", "Year"))
    profile_rows = _optional_rows(profile_path, ("Favorite Films",))
    diary_rows = _optional_rows(
        diary_path, ("Name", "Year", "Rating", "Rewatch", "Tags", "Watched Date")
    )
    watchlist = tuple(
        Film(
            watchlist_id=_required(row, "Letterboxd URI", watchlist_path),
            title=_required(row, "Name", watchlist_path),
            year=int(_required(row, "Year", watchlist_path)),
        )
        for row in watchlist_rows
    )
    ratings_tsv = _tsv(
        ("title", "year", "rating"),
        tuple((row["Name"], row["Year"], row["Rating"]) for row in ratings_rows),
    )
    reviews_tsv = _tsv(
        ("title", "year", "rating", "review", "tags"),
        tuple(
            (row["Name"], row["Year"], row["Rating"], row["Review"], row["Tags"])
            for row in reviews_rows
        ),
    )
    liked_films_tsv = _tsv(
        ("title", "year"),
        tuple((row["Name"], row["Year"]) for row in liked_films_rows),
    )
    watched_by_uri = {
        row["Letterboxd URI"].strip(): (row["Name"], row["Year"])
        for row in watched_rows
    }
    favorite_uris = (
        [item.strip() for item in profile_rows[0]["Favorite Films"].split(",")]
        if profile_rows
        else []
    )
    favorite_films_tsv = _tsv(
        ("title", "year"),
        tuple(watched_by_uri[uri] for uri in favorite_uris if uri in watched_by_uri),
    )
    ranking_rows = tuple(
        row
        for path in sorted((data_dir / "lists").glob("*.csv"))
        for row in _ranking_rows(path)
    )
    rankings_tsv = _tsv(("list", "position", "title", "year"), ranking_rows)
    diary_tsv = _tsv(
        ("watched_date", "title", "year", "rating", "rewatch", "tags"),
        tuple(
            (
                row["Watched Date"],
                row["Name"],
                row["Year"],
                row["Rating"],
                row["Rewatch"],
                row["Tags"],
            )
            for row in diary_rows
        ),
    )
    return LetterboxdData(
        watchlist=watchlist,
        ratings_tsv=ratings_tsv,
        reviews_tsv=reviews_tsv,
        liked_films_tsv=liked_films_tsv,
        favorite_films_tsv=favorite_films_tsv,
        rankings_tsv=rankings_tsv,
        diary_tsv=diary_tsv,
    )


def verified_catalog(data: LetterboxdData, cache: TmdbCache) -> str:
    rows = []
    # IDs curtos são mais fáceis para o modelo devolver sem alterar.
    candidate_id = 0
    for film in data.watchlist:
        fact = cache.movies.get(film.watchlist_id)
        if fact is None:
            continue
        candidate_id += 1
        rows.append(
            (
                candidate_id,
                fact.title,
                fact.year,
                ", ".join(fact.genres),
                ", ".join(fact.directors),
            )
        )
    return _tsv(("candidate_id", "title", "year", "genres", "directors"), rows)


def verified_candidates(data: LetterboxdData, cache: TmdbCache) -> dict[int, str]:
    ids = [film.watchlist_id for film in data.watchlist if film.watchlist_id in cache.movies]
    return {index: watchlist_id for index, watchlist_id in enumerate(ids, start=1)}


def _model_config(model: BaseChatModel) -> dict[str, object]:
    values = dict(model._identifying_params)
    values["type"] = model._llm_type
    dumped = model.model_dump(exclude_none=True)
    for field in (
        "model",
        "reasoning",
        "num_ctx",
        "num_predict",
        "temperature",
        "mirostat",
        "top_k",
        "top_p",
        "repeat_penalty",
        "seed",
        "format",
    ):
        if field in dumped:
            values[field] = dumped[field]
    return values


def _profile_refs(profile: TasteProfile) -> set[str]:
    return {
        reference
        for signal in (*profile.likes, *profile.dislikes)
        for reference in signal.evidence_ids
    }


def _validate_constraints(fact: object, constraints: SearchConstraints) -> None:
    if not hasattr(fact, "year") or not hasattr(fact, "runtime_minutes"):
        raise TypeError("invalid confirmed movie")
    year = fact.year
    runtime = fact.runtime_minutes
    if constraints.year_min is not None and year < constraints.year_min:
        raise ValueError("confirmed movie violates minimum year")
    if constraints.year_max is not None and year > constraints.year_max:
        raise ValueError("confirmed movie violates maximum year")
    if constraints.runtime_min is not None and (
        runtime is None or runtime < constraints.runtime_min
    ):
        raise ValueError("confirmed movie violates minimum runtime")
    if constraints.runtime_max is not None and (
        runtime is None or runtime > constraints.runtime_max
    ):
        raise ValueError("confirmed movie violates maximum runtime")


class Recommender:
    def __init__(
        self,
        data: LetterboxdData,
        model: BaseChatModel,
        cache: TmdbCache,
        taste_cache_path: Path | None = None,
        initialization_metrics: Mapping[str, object] | None = None,
    ) -> None:
        self.data = data
        self.model = model
        self.cache = cache
        self.evidence_store = EvidenceStore(data)
        self.taste_cache = TasteCache(taste_cache_path or Path("data/taste_profile.json"))
        self.initialization_metrics = dict(initialization_metrics or {})

    def _taste_profile(
        self, budget: RequestBudget
    ) -> tuple[TasteProfile, dict[str, object], bool, str | None]:
        key = self.taste_cache.fingerprint(
            self.data, _model_config(self.model), "evidence-profile-v2"
        )
        store = self.evidence_store
        profile = self.taste_cache.load(key)
        if profile is not None:
            references = _profile_refs(profile)
            if references.issubset(store.by_id):
                cached_evidence: dict[str, object] = {
                    reference: store.by_id[reference] for reference in references
                }
                return profile, cached_evidence, True, None
        state = TasteToolState()
        evidence: dict[str, object] = {}
        analyze = create_analyze_taste_tool(
            self.model, self.evidence_store, state, budget, evidence
        )
        analyze.invoke({})
        if state.profile is None:
            raise ValueError("taste specialist did not produce a profile")
        warning = self.taste_cache.save(key, state.profile)
        return state.profile, evidence, False, warning

    def recommend(
        self,
        request: str,
        flow_callback: Callable[[str], None] | None = None,
    ) -> CineResult:
        started = time.perf_counter()
        if not request.strip():
            raise ValueError("request must not be empty")
        available_ids = {
            film.watchlist_id
            for film in self.data.watchlist
            if film.watchlist_id in self.cache.movies
        }
        unavailable_count = len(self.data.watchlist) - len(available_ids)
        warnings: tuple[str, ...] = (
            (f"TMDb não confirmou {unavailable_count} filme(s) da watchlist.",)
            if unavailable_count
            else ()
        )
        if not available_ids:
            return CineResult(
                warnings=warnings or ("Nenhum filme foi confirmado pelo TMDb.",),
                metrics={**self.initialization_metrics, "total_seconds": time.perf_counter() - started},
            )
        budget = RequestBudget(flow_callback)
        constraints = request_constraints(request)
        profile, personal_evidence, cache_hit, cache_warning = self._taste_profile(budget)
        if cache_warning:
            warnings += (cache_warning,)
        taste_state = TasteToolState(profile=profile)
        analyze_taste = create_analyze_taste_tool(
            self.model, self.evidence_store, taste_state, budget, personal_evidence
        )
        history_state = HistoryToolState()
        analyze_history = create_analyze_history_tool(
            self.model,
            self.evidence_store,
            request,
            history_state,
            budget,
            personal_evidence,
        )
        catalog = LocalCatalog(self.data, self.cache)
        tmdb_state = TmdbToolState()
        get_tmdb_details = create_get_tmdb_details_tool(
            self.cache,
            verified_candidates(self.data, self.cache),
            tmdb_state,
        )
        recommendation_agent = create_recommendation_agent(
            self.model,
            [
                analyze_taste,
                analyze_history,
                create_search_candidates_tool(catalog, constraints),
                get_tmdb_details,
            ],
            [budget.scope("recommendation", model_limit=8, tool_limit=8)],
        )
        result = recommendation_agent.invoke(
            {
                "messages": [
                    recommendation_message(
                        request,
                        profile.model_dump(mode="json"),
                    )
                ]
            },
            config={"recursion_limit": 24, "max_concurrency": 1},
        )
        if not isinstance(result, Mapping):
            raise InvalidAgentResult("invalid recommendation agent result")
        choices = result.get("structured_response")
        if not isinstance(choices, RecommendationChoices):
            raise InvalidAgentResult("invalid recommendation agent structured response")
        # Os dados exibidos vêm do cache TMDb, não do texto gerado pela LLM.
        recommendations = []
        for choice in choices.recommendations:
            fact = tmdb_state.confirmed.get(choice.candidate_id)
            if fact is None:
                raise ValueError(
                    f"recommendation is not TMDb-confirmed: {choice.candidate_id}"
                )
            _validate_constraints(fact, constraints)
            invalid_personal = [
                reference
                for reference in choice.personal_evidence_ids
                if reference != "request" and reference not in personal_evidence
            ]
            if invalid_personal:
                raise ValueError(f"personal evidence was not supplied: {invalid_personal[0]}")
            if history_state.context is not None and not any(
                reference.startswith("diary:")
                for reference in choice.personal_evidence_ids
            ):
                raise ValueError("history-based recommendation must cite diary evidence")
            prefix = f"movie:{choice.candidate_id}:"
            invalid_movie = [
                reference
                for reference in choice.movie_evidence_ids
                if not reference.startswith(prefix)
                or reference not in catalog.supplied
                or reference not in catalog.evidence
            ]
            if invalid_movie:
                raise ValueError(f"movie evidence was not supplied: {invalid_movie[0]}")
            evidence = {
                reference: (
                    request if reference == "request" else personal_evidence[reference]
                )
                for reference in choice.personal_evidence_ids
            }
            evidence.update(
                {reference: catalog.evidence[reference] for reference in choice.movie_evidence_ids}
            )
            recommendations.append(
                Recommendation(
                    watchlist_id=fact.watchlist_id,
                    title=fact.title,
                    year=fact.year,
                    fit_reason=choice.fit_reason,
                    overview=fact.overview,
                    genres=fact.genres,
                    directors=fact.directors,
                    runtime_minutes=fact.runtime_minutes,
                    cast=fact.cast,
                    evidence=evidence,
                    is_primary=choice.candidate_id == choices.primary_candidate_id,
                )
            )
        recommendations.sort(key=lambda item: not item.is_primary)
        metrics = budget.snapshot()
        metrics.update(self.initialization_metrics)
        metrics["taste_cache_hit"] = cache_hit
        metrics["total_seconds"] = time.perf_counter() - started
        return CineResult(
            recommendations=tuple(recommendations),
            warnings=warnings,
            flow=tuple(budget.flow.events),
            metrics=metrics,
        )


class Recommends(Protocol):
    def recommend(
        self,
        request: str,
        flow_callback: Callable[[str], None] | None = None,
    ) -> CineResult: ...


def _render_evidence(
    recommendation: Recommendation, output_fn: Callable[[str], None]
) -> None:
    output_fn("EVIDÊNCIAS")
    output_fn("")
    for reference, evidence in recommendation.evidence.items():
        if reference == "request":
            output_fn(f"- pedido atual: {evidence}")
        elif reference.startswith("movie:"):
            output_fn(f"- {reference}: {evidence}")
        elif not isinstance(evidence, dict):
            output_fn(f"- {reference}: registro Letterboxd")
        else:
            fields = [
                str(evidence[field])
                for field in (
                    "watched_date",
                    "title",
                    "year",
                    "rating",
                    "review",
                    "tags",
                    "rewatch",
                    "list",
                    "position",
                )
                if evidence.get(field)
            ]
            output_fn(f"- {reference}: {' | '.join(fields)}")


def _render_movie(
    recommendation: Recommendation, output_fn: Callable[[str], None]
) -> None:
    output_fn(f"{recommendation.title} ({recommendation.year})")
    output_fn("")
    output_fn("Por que recomendo")
    output_fn("")
    output_fn(recommendation.fit_reason)
    output_fn("")
    _render_evidence(recommendation, output_fn)
    output_fn("")
    output_fn("FICHA DO FILME")
    output_fn("")
    output_fn(f"Direção: {', '.join(recommendation.directors) or 'não informada'}")
    output_fn(f"Sinopse: {recommendation.overview}")
    output_fn(f"Gêneros: {', '.join(recommendation.genres) or 'não informado'}")
    duration = (
        f"{recommendation.runtime_minutes} minutos"
        if recommendation.runtime_minutes is not None
        else "não informada"
    )
    output_fn(f"Duração: {duration}")
    output_fn(f"Elenco: {', '.join(recommendation.cast) or 'não informado'}")


def _render_result(result: CineResult, output_fn: Callable[[str], None]) -> None:
    if not result.recommendations:
        output_fn("Nenhuma recomendação válida encontrada.")
    else:
        primary = result.recommendations[0]
        output_fn("")
        output_fn("RECOMENDAÇÃO PRINCIPAL")
        output_fn("")
        _render_movie(primary, output_fn)
        alternatives = result.recommendations[1:]
        if alternatives:
            output_fn("")
            output_fn("OUTRAS OPÇÕES")
            for recommendation in alternatives:
                output_fn("")
                _render_movie(recommendation, output_fn)
    for warning in result.warnings:
        output_fn(f"Aviso: {warning}")


def run_cli(
    recommender: Recommends,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    output_fn("Olá, Michael! Gostaria de uma recomendação de filme da sua watchlist hoje?")
    request = input_fn("Conte o que você está procurando: ").strip()
    if not request:
        output_fn("Nenhum pedido informado.")
        return
    flow_started = False

    def show_flow(event: str) -> None:
        nonlocal flow_started
        if not flow_started:
            output_fn("")
            output_fn("FLUXO DOS AGENTES")
            output_fn("")
            flow_started = True
        output_fn(event)

    _render_result(
        recommender.recommend(request, flow_callback=show_flow),
        output_fn,
    )


def tmdb_token() -> str:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    token = os.getenv("TMDB_API_TOKEN")
    if not token:
        raise RuntimeError("TMDB_API_TOKEN não foi definido no arquivo .env")
    return token


def create_recommender() -> Recommender:
    started = time.perf_counter()
    data_dir = Path(os.getenv("CINE_DATA_DIR", "data"))
    data = load_letterboxd(data_dir)
    letterboxd_loaded = time.perf_counter()
    cache_path = data_dir / "tmdb_cache.json"
    client = TmdbClient(token=tmdb_token())
    try:
        cache = sync_tmdb_cache(
            data.watchlist,
            load_tmdb_cache(cache_path),
            client,
            cache_path,
        )
    finally:
        client.close()
    tmdb_synced = time.perf_counter()
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        num_ctx=32768,
        num_predict=1024,
        reasoning=False,
        temperature=0,
    )
    return Recommender(
        data,
        model,
        cache,
        taste_cache_path=data_dir / "taste_profile.json",
        initialization_metrics={
            "letterboxd_load_seconds": letterboxd_loaded - started,
            "tmdb_sync_seconds": tmdb_synced - letterboxd_loaded,
            "initialization_seconds": time.perf_counter() - started,
        },
    )


def main() -> None:
    run_cli(create_recommender())


if __name__ == "__main__":
    main()
