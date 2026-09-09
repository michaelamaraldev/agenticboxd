from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import ClassVar

from langchain.tools import tool
from pydantic import ValidationError

from .models import LetterboxdData, TasteProfile, TmdbCache


@dataclass(frozen=True, slots=True)
class SearchConstraints:
    year_min: int | None = None
    year_max: int | None = None
    runtime_min: int | None = None
    runtime_max: int | None = None


def request_constraints(request: str) -> SearchConstraints:
    text = normalize(request)
    runtime_max = _number_after(
        text,
        r"(?:ate|no maximo|menos de|maximo de)\s+(\d+)\s*(?:minutos|min\b)",
    )
    runtime_min = _number_after(
        text,
        r"(?:pelo menos|no minimo|mais de|minimo de)\s+(\d+)\s*(?:minutos|min\b)",
    )
    year_min = _number_after(
        text,
        r"(?:lancad[oa]\s+)?(?:depois de|a partir de)\s+(\d{4})\b",
    )
    year_max = _number_after(
        text,
        r"(?:lancad[oa]\s+)?(?:antes de|ate)\s+(\d{4})\b",
    )
    return SearchConstraints(year_min, year_max, runtime_min, runtime_max)


def _number_after(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def normalize(value: object) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", str(value).casefold())
        if not unicodedata.combining(character)
    )


def page_result(
    records: list[dict[str, object]], page: int, page_size: int
) -> dict[str, object]:
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be positive")
    page_size = min(page_size, 20)
    start = (page - 1) * page_size
    selected = records[start : start + page_size]
    return {
        "records": selected,
        "total": len(records),
        "page": page,
        "page_size": page_size,
        "has_more": start + len(selected) < len(records),
        "coverage": "partial" if len(selected) < len(records) else "complete",
    }


class LocalCatalog:
    def __init__(self, data: LetterboxdData, cache: TmdbCache):
        watchlist_ids = dict.fromkeys(
            film.watchlist_id
            for film in data.watchlist
            if film.watchlist_id in cache.movies
        )
        self.candidates = dict(enumerate(watchlist_ids, 1))
        self.records: list[dict[str, object]] = []
        self.evidence: dict[str, object] = {}
        self.supplied: set[str] = set()
        for candidate_id, watchlist_id in self.candidates.items():
            record: dict[str, object] = {
                "candidate_id": candidate_id,
                **cache.movies[watchlist_id].model_dump(mode="json"),
            }
            references: dict[str, object] = {}
            for field in (
                "title",
                "year",
                "overview",
                "genres",
                "directors",
                "cast",
                "runtime_minutes",
            ):
                value = record[field]
                if value is not None and value != "" and value != []:
                    references[f"movie:{candidate_id}:{field}"] = value
            record["evidence_ids"] = list(references)
            self.evidence.update(references)
            self.records.append(record)

    def search(
        self,
        query: str = "",
        page: int = 1,
        page_size: int = 10,
        year_min: int | None = None,
        year_max: int | None = None,
        runtime_min: int | None = None,
        runtime_max: int | None = None,
    ) -> dict[str, object]:
        terms = normalize(query).split()
        matches: list[dict[str, object]] = []
        scored: list[tuple[int, dict[str, object]]] = []
        for record in self.records:
            searchable = normalize(
                " ".join(
                    str(record[field])
                    for field in ("title", "overview", "genres", "directors", "cast")
                )
            )
            score = sum(term in searchable for term in terms)
            if terms and score == 0:
                continue
            year = record["year"]
            if not isinstance(year, int):
                continue
            if year_min is not None and year < year_min:
                continue
            if year_max is not None and year > year_max:
                continue
            runtime = record["runtime_minutes"]
            if runtime_min is not None and (
                not isinstance(runtime, int) or runtime < runtime_min
            ):
                continue
            if runtime_max is not None and (
                not isinstance(runtime, int) or runtime > runtime_max
            ):
                continue
            scored.append((score, record))
        matches = [record for _, record in sorted(scored, key=lambda item: -item[0])]
        result = page_result(matches, page, page_size)
        selected = result["records"]
        if isinstance(selected, list):
            for record in selected:
                if not isinstance(record, dict):
                    continue
                references = record.get("evidence_ids")
                if isinstance(references, list):
                    self.supplied.update(str(reference) for reference in references)
        return result


class EvidenceStore:
    sources: ClassVar[dict[str, str]] = {
        "ratings": "ratings_tsv",
        "reviews": "reviews_tsv",
        "likes": "liked_films_tsv",
        "favorites": "favorite_films_tsv",
        "rankings": "rankings_tsv",
        "diary": "diary_tsv",
    }

    def __init__(self, data: LetterboxdData):
        self.records: dict[str, list[dict[str, object]]] = {}
        self.by_id: dict[str, dict[str, object]] = {}
        self.supplied: set[str] = set()
        for source, attribute in self.sources.items():
            rows = csv.DictReader(
                io.StringIO(getattr(data, attribute)), delimiter="\t"
            )
            self.records[source] = []
            for index, row in enumerate(rows, 1):
                reference = f"{source}:{index}"
                record: dict[str, object] = {"evidence_id": reference, **row}
                self.records[source].append(record)
                self.by_id[reference] = record

    def fork(self) -> EvidenceStore:
        store = object.__new__(EvidenceStore)
        store.records = self.records
        store.by_id = self.by_id
        store.supplied = set()
        return store

    def summary(self, sources: tuple[str, ...]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for source in sources:
            rows = self.records[source]
            details: dict[str, object] = {"total": len(rows)}
            if source in ("ratings", "diary"):
                details["ratings"] = dict(
                    Counter(str(row.get("rating", "")) for row in rows)
                )
            if source == "diary":
                dates = sorted(
                    str(row["watched_date"])
                    for row in rows
                    if row.get("watched_date")
                )
                details["date_range"] = [dates[0], dates[-1]] if dates else []
            result[source] = details
        return result

    def search(
        self,
        source: str,
        query: str = "",
        page: int = 1,
        page_size: int = 10,
        date_from: str | None = None,
        date_to: str | None = None,
        rewatch: bool | None = None,
    ) -> dict[str, object]:
        if source not in self.records:
            raise ValueError("unknown evidence source")
        for value in (date_from, date_to):
            if value is not None:
                date.fromisoformat(value)
        terms = normalize(query).split()
        matches: list[dict[str, object]] = []
        for record in self.records[source]:
            searchable = normalize(" ".join(str(value) for value in record.values()))
            if not all(term in searchable for term in terms):
                continue
            watched = str(record.get("watched_date", ""))
            if date_from and watched < date_from:
                continue
            if date_to and (not watched or watched > date_to):
                continue
            is_rewatch = normalize(record.get("rewatch", "")) in ("yes", "true", "1")
            if rewatch is not None and is_rewatch != rewatch:
                continue
            matches.append(record)
        result = page_result(matches, page, page_size)
        selected = result["records"]
        if isinstance(selected, list):
            self.supplied.update(
                str(record["evidence_id"])
                for record in selected
                if isinstance(record, dict) and "evidence_id" in record
            )
        return result

    def validate_refs(self, references: set[str] | tuple[str, ...] | list[str]) -> None:
        if any(
            reference not in self.supplied or reference not in self.by_id
            for reference in references
        ):
            raise ValueError("Evidence reference was not supplied or does not exist")


class TasteCache:
    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def fingerprint(data, config: dict, version: str) -> str:
        sources = {
            field: getattr(data, field)
            for field in (
                "ratings_tsv",
                "reviews_tsv",
                "liked_films_tsv",
                "favorite_films_tsv",
                "rankings_tsv",
            )
        }
        payload = json.dumps(
            {"sources": sources, "config": config, "version": version},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, key: str) -> TasteProfile | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("key") != key:
                return None
            return TasteProfile.model_validate(payload["profile"])
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            return None

    def save(self, key: str, profile: TasteProfile) -> str | None:
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    {"key": key, "profile": profile.model_dump(mode="json")},
                    handle,
                    ensure_ascii=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            return None
        except OSError:
            return (
                "Não foi possível salvar o perfil local; a análise continua "
                "disponível neste pedido."
            )
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)


def create_search_candidates_tool(
    catalog: LocalCatalog, constraints: SearchConstraints | None = None
):
    constraints = constraints or SearchConstraints()

    @tool
    def search_candidates(
        query: str = "",
        page: int = 1,
        page_size: int = 10,
        year_min: int | None = None,
        year_max: int | None = None,
        runtime_min: int | None = None,
        runtime_max: int | None = None,
    ) -> str:
        """Search verified watchlist by lexical terms and year/runtime; empty query browses pages of at most 20."""
        effective_year_min = (
            max(
                value
                for value in (year_min, constraints.year_min)
                if value is not None
            )
            if year_min is not None or constraints.year_min is not None
            else None
        )
        effective_year_max = (
            min(
                value
                for value in (year_max, constraints.year_max)
                if value is not None
            )
            if year_max is not None or constraints.year_max is not None
            else None
        )
        effective_runtime_min = (
            max(
                value
                for value in (runtime_min, constraints.runtime_min)
                if value is not None
            )
            if runtime_min is not None or constraints.runtime_min is not None
            else None
        )
        effective_runtime_max = (
            min(
                value
                for value in (runtime_max, constraints.runtime_max)
                if value is not None
            )
            if runtime_max is not None or constraints.runtime_max is not None
            else None
        )
        return json.dumps(
            catalog.search(
                query,
                page,
                min(page_size, 10),
                effective_year_min,
                effective_year_max,
                effective_runtime_min,
                effective_runtime_max,
            ),
            ensure_ascii=False,
        )

    return search_candidates


def create_taste_search_tool(store: EvidenceStore):
    @tool
    def search_taste_evidence(
        source: str, query: str = "", page: int = 1, page_size: int = 10
    ) -> str:
        """Read one taste source; use an empty query to browse and words only to match record contents."""
        if source not in ("ratings", "reviews", "likes", "favorites", "rankings"):
            raise ValueError("Taste can only access taste sources")
        return json.dumps(
            store.search(source, query, page, page_size), ensure_ascii=False
        )

    return search_taste_evidence


def create_history_search_tool(store: EvidenceStore):
    @tool
    def search_history_evidence(
        query: str = "",
        page: int = 1,
        page_size: int = 10,
        date_from: str | None = None,
        date_to: str | None = None,
        rewatch: bool | None = None,
    ) -> str:
        """Read diary evidence by text, ISO dates and rewatch flag; results have IDs and partial coverage."""
        return json.dumps(
            store.search(
                "diary",
                query,
                page,
                min(page_size, 20),
                date_from,
                date_to,
                rewatch,
            ),
            ensure_ascii=False,
        )

    return search_history_evidence
