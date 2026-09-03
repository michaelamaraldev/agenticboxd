from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_ollama import ChatOllama

from app import Recommender, load_letterboxd, tmdb_token
from models import Film, LetterboxdData, TmdbCache, TmdbFact
from tmdb import TmdbClient, load_tmdb_cache

SEMANTIC_REQUESTS = (
    "Me recomende um filme de romance parecido com os filmes do Rohmer.",
    "Quero um terror psicológico sem gore, com uma atmosfera claustrofóbica.",
    "Quero uma comédia leve e acolhedora para assistir depois de um dia cansativo.",
    "Me recomende uma ficção científica contemplativa sobre solidão.",
    "Quero algo formalmente ousado, mas emocionalmente contido.",
)


def local_model() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        num_ctx=32768,
        num_predict=1024,
        reasoning=False,
        temperature=0,
    )


def fake_cache(films: tuple[Film, ...]) -> TmdbCache:
    return TmdbCache(
        movies={
            film.watchlist_id: TmdbFact(
                watchlist_id=film.watchlist_id,
                tmdb_id=index,
                title=film.title,
                year=film.year,
                overview="Confirmed overview.",
                genres=("Drama",),
                directors=("Director",),
                runtime_minutes=100,
                cast=(),
                release_date=None,
            )
            for index, film in enumerate(films)
        }
    )


@pytest.mark.ollama
@pytest.mark.parametrize("target_index", [0, 586, 1173])
def test_qwen_can_choose_any_position_from_verified_catalog(target_index: int) -> None:
    if os.getenv("CINE_RUN_OLLAMA") != "1":
        pytest.skip("defina CINE_RUN_OLLAMA=1 para habilitar a integração local")
    films = tuple(
        Film(
            watchlist_id=f"film:{index}",
            title="EXACT TARGET" if index == target_index else f"Movie {index:04d}",
            year=2000,
        )
        for index in range(1174)
    )
    data = LetterboxdData(
        watchlist=films,
        ratings_tsv="title\tyear\trating\n",
        reviews_tsv="title\tyear\trating\treview\ttags\n",
    )

    result = Recommender(data, local_model(), fake_cache(films)).recommend(
        "Recommend only the movie named EXACT TARGET."
    )

    assert any(item.watchlist_id == f"film:{target_index}" for item in result.recommendations)


@pytest.mark.tmdb
def test_live_tmdb_dotenv_token() -> None:
    if os.getenv("CINE_RUN_TMDB") != "1":
        pytest.skip("defina CINE_RUN_TMDB=1 para habilitar a integração real")

    fact = TmdbClient(token=tmdb_token()).enrich("film:the-matrix", "The Matrix", 1999)

    assert fact is not None
    assert fact.tmdb_id > 0


@pytest.mark.ollama
@pytest.mark.tmdb
def test_semantic_requests_with_real_cache(real_data_dir: Path) -> None:
    if os.getenv("CINE_RUN_OLLAMA") != "1" or os.getenv("CINE_RUN_TMDB") != "1":
        pytest.skip("habilite as integrações Ollama e TMDb")
    data = load_letterboxd(real_data_dir)
    cache = load_tmdb_cache(real_data_dir / "tmdb_cache.json")
    watchlist_ids = {film.watchlist_id for film in data.watchlist}
    recommender = Recommender(data, local_model(), cache)

    for request in SEMANTIC_REQUESTS:
        result = recommender.recommend(request)

        assert 1 <= len(result.recommendations) <= 3, request
        ids = [item.watchlist_id for item in result.recommendations]
        assert len(ids) == len(set(ids)), request
        assert set(ids) <= watchlist_ids, request
        for recommendation in result.recommendations:
            fact = cache.movies[recommendation.watchlist_id]
            assert recommendation.title == fact.title, request
            assert recommendation.year == fact.year, request
            assert recommendation.overview == fact.overview, request
            assert recommendation.genres == fact.genres, request
            assert recommendation.directors == fact.directors, request
            assert recommendation.runtime_minutes == fact.runtime_minutes, request
            assert recommendation.cast == fact.cast, request
            assert recommendation.fit_reason.strip(), request
        print(f"{request} -> {', '.join(item.title for item in result.recommendations)}")
