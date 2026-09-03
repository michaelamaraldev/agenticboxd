from __future__ import annotations

import argparse
import csv
import io
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.agents.recommendation import (
    create_recommendation_agent,
    final_prompt,
    final_request,
    selection_prompt,
    selection_request,
)
from src.agents.taste import create_taste_agent
from src.models import (
    CineResult,
    Film,
    LetterboxdData,
    Recommendation,
    RecommendationChoices,
    TmdbCache,
    WatchlistSelection,
)
from src.tmdb import TmdbClient, load_tmdb_cache, sync_tmdb_cache
from src.tools.analyze_taste import TasteToolState, create_analyze_taste_tool
from src.tools.get_tmdb_details import TmdbToolState, create_get_tmdb_details_tool


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


def load_letterboxd(data_dir: Path) -> LetterboxdData:
    watchlist_path = data_dir / "watchlist.csv"
    ratings_path = data_dir / "ratings.csv"
    reviews_path = data_dir / "reviews.csv"
    watchlist_rows = _rows(watchlist_path, ("Name", "Year", "Letterboxd URI"))
    ratings_rows = _rows(ratings_path, ("Name", "Year", "Rating"))
    reviews_rows = _rows(reviews_path, ("Name", "Year", "Rating", "Review", "Tags"))
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
    return LetterboxdData(
        watchlist=watchlist,
        ratings_tsv=ratings_tsv,
        reviews_tsv=reviews_tsv,
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


class Recommender:
    def __init__(
        self,
        data: LetterboxdData,
        model: BaseChatModel,
        cache: TmdbCache,
    ) -> None:
        self.data = data
        self.model = model
        self.cache = cache
        self.taste_agent = create_taste_agent(model)
        self.taste_state = TasteToolState()
        self.analyze_taste = create_analyze_taste_tool(
            self.taste_agent, data, self.taste_state
        )

    def recommend(self, request: str) -> CineResult:
        if not request.strip():
            raise ValueError("request must not be empty")
        available_ids = {
            film.watchlist_id
            for film in self.data.watchlist
            if film.watchlist_id in self.cache.movies
        }
        unavailable_count = len(self.data.watchlist) - len(available_ids)
        warnings = (
            (f"TMDb não confirmou {unavailable_count} filme(s) da watchlist.",)
            if unavailable_count
            else ()
        )
        if not available_ids:
            return CineResult(warnings=warnings or ("Nenhum filme foi confirmado pelo TMDb.",))
        taste_profile = self.analyze_taste.invoke({})
        if not isinstance(taste_profile, str):
            raise ValueError("invalid analyze_taste result")
        # Primeiro, a LLM escolhe até dez filmes olhando o catálogo completo.
        selection_agent = create_recommendation_agent(
            self.model,
            selection_prompt(),
            WatchlistSelection,
        )
        selection_result = selection_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": selection_request(
                            request,
                            taste_profile,
                            verified_catalog(self.data, self.cache),
                        ),
                    }
                ]
            }
        )
        if not isinstance(selection_result, Mapping):
            raise ValueError("invalid selection agent result")
        selection = selection_result.get("structured_response")
        if not isinstance(selection, WatchlistSelection):
            raise ValueError("invalid selection agent structured response")
        # Só os escolhidos recebem os detalhes completos do TMDb.
        tmdb_state = TmdbToolState()
        get_tmdb_details = create_get_tmdb_details_tool(
            self.cache,
            verified_candidates(self.data, self.cache),
            tmdb_state,
        )
        details = get_tmdb_details.invoke({"candidate_ids": selection.candidate_ids})
        if not isinstance(details, str):
            raise ValueError("invalid get_tmdb_details result")
        recommendation_agent = create_recommendation_agent(
            self.model,
            final_prompt(),
            RecommendationChoices,
        )
        result = recommendation_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": final_request(request, taste_profile, details),
                    }
                ]
            }
        )
        if not isinstance(result, Mapping):
            raise ValueError("invalid recommendation agent result")
        choices = result.get("structured_response")
        if not isinstance(choices, RecommendationChoices):
            raise ValueError("invalid recommendation agent structured response")
        # Os dados exibidos vêm do cache TMDb, não do texto gerado pela LLM.
        recommendations = []
        for choice in choices.recommendations:
            fact = tmdb_state.confirmed.get(choice.selection_position)
            if fact is None:
                raise ValueError(
                    f"recommendation is not TMDb-confirmed: {choice.selection_position}"
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
                )
            )
        return CineResult(recommendations=tuple(recommendations), warnings=warnings)


class Recommends(Protocol):
    def recommend(self, request: str) -> CineResult: ...


def _render_result(result: CineResult, output_fn: Callable[[str], None]) -> None:
    if not result.recommendations:
        output_fn("Nenhuma recomendação válida encontrada.")
    for index, recommendation in enumerate(result.recommendations, start=1):
        output_fn(f"{index}. {recommendation.title} ({recommendation.year})")
        output_fn(recommendation.fit_reason)
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
    for warning in result.warnings:
        output_fn(f"Aviso: {warning}")


def run_cli(
    recommender: Recommends,
    utterance: str | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    request = utterance if utterance is not None else input_fn("O que você quer assistir? ")
    _render_result(recommender.recommend(request.strip()), output_fn)


def tmdb_token() -> str:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    token = os.getenv("TMDB_API_TOKEN")
    if not token:
        raise RuntimeError("TMDB_API_TOKEN não foi definido no arquivo .env")
    return token


def create_recommender() -> Recommender:
    data_dir = Path(os.getenv("CINE_DATA_DIR", "data"))
    data = load_letterboxd(data_dir)
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
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        num_ctx=32768,
        num_predict=1024,
        reasoning=False,
        temperature=0,
    )
    return Recommender(data, model, cache)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recomenda filmes da sua watchlist do Letterboxd com dados confirmados pelo TMDb."
        )
    )
    parser.add_argument(
        "--utterance",
        help='Pedido em linguagem natural. Exemplo: "Me recomende um romance parecido com Rohmer."',
    )
    arguments = parser.parse_args(argv)
    run_cli(create_recommender(), utterance=arguments.utterance)


if __name__ == "__main__":
    main()
