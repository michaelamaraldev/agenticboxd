from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Film(BaseModel):
    model_config = ConfigDict(frozen=True)

    watchlist_id: str
    title: str
    year: int


@dataclass(frozen=True, slots=True)
class LetterboxdData:
    watchlist: tuple[Film, ...]
    ratings_tsv: str
    reviews_tsv: str
    liked_films_tsv: str = ""
    favorite_films_tsv: str = ""
    rankings_tsv: str = ""
    diary_tsv: str = ""


class EvidenceSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=600)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=10)

    @field_validator("text", mode="before")
    @classmethod
    def limit_text(cls, value: object) -> object:
        return value[:600] if isinstance(value, str) else value

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def limit_evidence_ids(cls, value: object) -> object:
        return value[:10] if isinstance(value, (list, tuple)) else value


class TasteProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    likes: tuple[EvidenceSignal, ...] = Field(default=(), max_length=5)
    dislikes: tuple[EvidenceSignal, ...] = Field(default=(), max_length=5)


class HistoryContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    relevant_patterns: tuple[EvidenceSignal, ...] = Field(default=(), max_length=5)
    rewatch_signals: tuple[EvidenceSignal, ...] = Field(default=(), max_length=3)

    @field_validator("relevant_patterns", mode="before")
    @classmethod
    def limit_patterns(cls, value: object) -> object:
        return value[:5] if isinstance(value, (list, tuple)) else value

    @field_validator("rewatch_signals", mode="before")
    @classmethod
    def limit_rewatch_signals(cls, value: object) -> object:
        return value[:3] if isinstance(value, (list, tuple)) else value


class WatchlistSelection(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_ids: tuple[int, ...] = Field(
        min_length=1,
        max_length=10,
        description="Unique candidate_ids copied from the verified watchlist catalog.",
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> WatchlistSelection:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        return self


class RecommendationChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: int = Field(
        ge=1,
        description=(
            "Copy the candidate_id from one movie returned by get_tmdb_details. "
            "This is not a TMDb ID."
        ),
    )
    fit_reason: str = Field(
        min_length=1,
        description="A concise subjective reason written in Brazilian Portuguese.",
    )
    personal_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=10)
    movie_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=10)


class RecommendationChoices(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary_candidate_id: int | None = None
    recommendations: tuple[RecommendationChoice, ...] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> RecommendationChoices:
        ids = [item.candidate_id for item in self.recommendations]
        if len(ids) != len(set(ids)):
            raise ValueError("recommendation candidate_ids must be unique")
        if ids and self.primary_candidate_id is None:
            raise ValueError("nonempty recommendations require a primary_candidate_id")
        if not ids and self.primary_candidate_id is not None:
            raise ValueError("empty recommendations require primary_candidate_id None")
        if self.primary_candidate_id is not None and self.primary_candidate_id not in ids:
            raise ValueError("primary_candidate_id must identify a recommendation")
        return self


class TmdbFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    watchlist_id: str
    tmdb_id: int
    title: str
    year: int
    overview: str
    genres: tuple[str, ...]
    directors: tuple[str, ...]
    runtime_minutes: int | None
    cast: tuple[str, ...]
    release_date: date | None


class TmdbCache(BaseModel):
    model_config = ConfigDict(frozen=True)

    movies: dict[str, TmdbFact] = Field(default_factory=dict)
    unmatched: tuple[str, ...] = ()


class Recommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    watchlist_id: str
    title: str
    year: int
    fit_reason: str
    overview: str
    genres: tuple[str, ...]
    directors: tuple[str, ...]
    runtime_minutes: int | None
    cast: tuple[str, ...]
    evidence: dict[str, object] = Field(default_factory=dict)
    is_primary: bool = False


class CineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    recommendations: tuple[Recommendation, ...] = ()
    warnings: tuple[str, ...] = ()
    flow: tuple[str, ...] = ()
    metrics: dict[str, object] = Field(default_factory=dict)
