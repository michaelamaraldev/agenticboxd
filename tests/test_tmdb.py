from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from src.models import Film, TmdbCache, TmdbFact
from src.tmdb import (
    TmdbClient,
    load_tmdb_cache,
    save_tmdb_cache,
    select_unique_match,
    sync_tmdb_cache,
)


def test_match_accepts_exact_title_or_original_title_and_year() -> None:
    candidates = [
        {
            "id": 1,
            "title": "Another",
            "original_title": "Conte d'été",
            "release_date": "1996-06-05",
        }
    ]

    assert select_unique_match("Conte d'été", 1996, candidates) == candidates[0]
    assert select_unique_match("Conte d'été", 1995, candidates) is None


def test_match_rejects_ambiguous_results() -> None:
    candidates = [
        {"id": 1, "title": "The Film", "release_date": "2000-01-01"},
        {"id": 2, "title": "The Film", "release_date": "2000-09-01"},
    ]

    assert select_unique_match("The Film", 2000, candidates) is None


def test_tmdb_client_fetches_directors_and_full_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search/movie"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"id": 42, "title": "The Film", "release_date": "2000-01-02"}
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "title": "The Film",
                "release_date": "2000-01-02",
                "overview": "Verified overview.",
                "runtime": 97,
                "genres": [{"name": "Romance"}],
                "credits": {
                    "cast": [{"name": "Actor One"}],
                    "crew": [{"name": "Director One", "job": "Director"}],
                },
            },
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.themoviedb.org/3/"
    )

    fact = TmdbClient("token", http_client=http_client).enrich(
        "https://boxd.it/key", "The Film", 2000
    )

    assert fact is not None
    assert fact.watchlist_id == "https://boxd.it/key"
    assert fact.directors == ("Director One",)
    assert fact.genres == ("Romance",)
    assert fact.cast == ("Actor One",)


class FakeTmdb:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enrich(self, watchlist_id: str, title: str, year: int) -> TmdbFact | None:
        self.calls.append(watchlist_id)
        if title == "Missing":
            return None
        return TmdbFact(
            watchlist_id=watchlist_id,
            tmdb_id=year,
            title=title,
            year=year,
            overview="Overview",
            genres=("Drama",),
            directors=("Director",),
            runtime_minutes=100,
            cast=(),
            release_date=None,
        )


def test_cache_is_persistent_and_syncs_only_new_movies(tmp_path: Path) -> None:
    cache_path = tmp_path / "tmdb_cache.json"
    films = (
        Film(watchlist_id="a", title="Found", year=2000),
        Film(watchlist_id="b", title="Missing", year=2001),
    )
    client = FakeTmdb()

    first = sync_tmdb_cache(films, TmdbCache(), client, cache_path)
    second = sync_tmdb_cache(films, load_tmdb_cache(cache_path), client, cache_path)

    assert client.calls == ["a", "b"]
    assert set(first.movies) == {"a"}
    assert first.unmatched == ("b",)
    assert second == first
    payload: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["unmatched"] == ["b"]


def test_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache = TmdbCache(unmatched=("missing",))

    save_tmdb_cache(path, cache)

    assert load_tmdb_cache(path) == cache
