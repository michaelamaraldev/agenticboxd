from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from langchain.tools import tool
from langchain_core.tools import BaseTool

from ..models import TmdbCache, TmdbFact, WatchlistSelection


@dataclass(slots=True)
class TmdbToolState:
    selected_ids: tuple[int, ...] | None = None
    confirmed: dict[int, TmdbFact] = field(default_factory=dict)


def create_get_tmdb_details_tool(
    cache: TmdbCache,
    candidates: Mapping[int, str],
    state: TmdbToolState,
) -> BaseTool:
    """Build the TMDb-details tool used by the recommendation agent."""

    @tool(args_schema=WatchlistSelection)
    def get_tmdb_details(candidate_ids: tuple[int, ...]) -> str:
        """Return verified TMDb details for up to 10 selected numeric candidate IDs."""
        unknown = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates]
        if unknown:
            raise ValueError(f"unknown candidate_id: {unknown[0]}")
        if state.selected_ids is not None and state.selected_ids != candidate_ids:
            raise ValueError("watchlist selection is already defined")
        state.selected_ids = candidate_ids
        state.confirmed = {
            position: cache.movies[candidates[candidate_id]]
            for position, candidate_id in enumerate(candidate_ids, start=1)
        }
        return json.dumps(
            {
                "movies": {
                    selection_position: fact.model_dump(mode="json")
                    for selection_position, fact in state.confirmed.items()
                }
            },
            ensure_ascii=False,
        )

    return get_tmdb_details
