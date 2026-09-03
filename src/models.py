from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class TasteProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    likes: tuple[str, ...]
    dislikes: tuple[str, ...]


class HistoryContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    relevant_patterns: tuple[str, ...] = Field(min_length=1, max_length=5)
    rewatch_signals: tuple[str, ...] = Field(default=(), max_length=3)


class WatchlistSelection(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_ids: tuple[int, ...] = Field(max_length=10)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> WatchlistSelection:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        return self


class RecommendationChoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    selection_position: int
    fit_reason: str = Field(min_length=1)


class RecommendationChoices(BaseModel):
    model_config = ConfigDict(frozen=True)

    recommendations: tuple[RecommendationChoice, ...] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> RecommendationChoices:
        ids = [item.selection_position for item in self.recommendations]
        if len(ids) != len(set(ids)):
            raise ValueError("recommendation selection_positions must be unique")
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


class CineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    recommendations: tuple[Recommendation, ...] = ()
    warnings: tuple[str, ...] = ()
