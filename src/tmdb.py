from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import httpx

from .models import Film, TmdbCache, TmdbFact


def normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold().strip())


def _release_year(candidate: Mapping[str, Any]) -> int | None:
    release_date = candidate.get("release_date")
    if not isinstance(release_date, str) or len(release_date) < 4:
        return None
    try:
        return int(release_date[:4])
    except ValueError:
        return None


def select_unique_match(
    title: str, year: int, candidates: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    expected = normalize_title(title)
    matches = []
    for candidate in candidates:
        titles = {
            normalize_title(value)
            for value in (candidate.get("title"), candidate.get("original_title"))
            if isinstance(value, str)
        }
        if expected in titles and _release_year(candidate) == year:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


class TmdbClient:
    def __init__(
        self,
        token: str,
        http_client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url="https://api.themoviedb.org/3/",
            headers={"Authorization": f"Bearer {token}", "accept": "application/json"},
            timeout=timeout,
        )

    def enrich(self, watchlist_id: str, title: str, year: int) -> TmdbFact | None:
        search = self._http.get(
            "search/movie",
            params={
                "query": title,
                "year": year,
                "language": "en-US",
                "include_adult": "false",
            },
        )
        search.raise_for_status()
        payload = search.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not isinstance(results, list):
            return None
        match = select_unique_match(title, year, results)
        if match is None or not isinstance(match.get("id"), int):
            return None
        tmdb_id = match["id"]
        response = self._http.get(
            f"movie/{tmdb_id}",
            params={"language": "pt-BR", "append_to_response": "credits"},
        )
        response.raise_for_status()
        details = response.json()
        if not isinstance(details, dict):
            return None
        return self._fact(watchlist_id, title, year, tmdb_id, details)

    def _fact(
        self,
        watchlist_id: str,
        fallback_title: str,
        fallback_year: int,
        tmdb_id: int,
        details: Mapping[str, Any],
    ) -> TmdbFact:
        release_date = None
        release_value = details.get("release_date")
        if isinstance(release_value, str) and release_value:
            with suppress(ValueError):
                release_date = date.fromisoformat(release_value)
        genres_payload = details.get("genres")
        genres = (
            tuple(
                item["name"]
                for item in genres_payload
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            )
            if isinstance(genres_payload, list)
            else ()
        )
        credits = details.get("credits")
        cast_payload = credits.get("cast") if isinstance(credits, dict) else None
        crew_payload = credits.get("crew") if isinstance(credits, dict) else None
        cast = (
            tuple(
                item["name"]
                for item in cast_payload[:5]
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            )
            if isinstance(cast_payload, list)
            else ()
        )
        directors = (
            tuple(
                item["name"]
                for item in crew_payload
                if isinstance(item, dict)
                and item.get("job") == "Director"
                and isinstance(item.get("name"), str)
            )
            if isinstance(crew_payload, list)
            else ()
        )
        title = details.get("title")
        overview = details.get("overview")
        runtime = details.get("runtime")
        return TmdbFact(
            watchlist_id=watchlist_id,
            tmdb_id=tmdb_id,
            title=title if isinstance(title, str) and title else fallback_title,
            year=release_date.year if release_date is not None else fallback_year,
            overview=overview if isinstance(overview, str) else "",
            genres=genres,
            directors=directors,
            runtime_minutes=runtime if isinstance(runtime, int) else None,
            cast=cast,
            release_date=release_date,
        )

    def close(self) -> None:
        if self._owns_client:
            self._http.close()


class MovieEnricher(Protocol):
    def enrich(self, watchlist_id: str, title: str, year: int) -> TmdbFact | None: ...


def load_tmdb_cache(path: Path) -> TmdbCache:
    if not path.exists():
        return TmdbCache()
    return TmdbCache.model_validate_json(path.read_text(encoding="utf-8"))


def save_tmdb_cache(path: Path, cache: TmdbCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cache.model_dump_json(indent=2), encoding="utf-8")


def sync_tmdb_cache(
    films: Sequence[Film],
    cache: TmdbCache,
    client: MovieEnricher,
    path: Path,
) -> TmdbCache:
    movies = dict(cache.movies)
    unmatched = list(cache.unmatched)
    known = set(movies) | set(unmatched)
    changed = False
    for film in films:
        if film.watchlist_id in known:
            continue
        fact = client.enrich(film.watchlist_id, film.title, film.year)
        if fact is None:
            unmatched.append(film.watchlist_id)
        else:
            movies[film.watchlist_id] = fact
        known.add(film.watchlist_id)
        changed = True
    result = TmdbCache(movies=movies, unmatched=tuple(unmatched))
    if changed or not path.exists():
        save_tmdb_cache(path, result)
    return result
