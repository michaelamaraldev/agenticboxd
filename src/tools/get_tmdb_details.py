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
        """Call once with up to 10 investigated candidate IDs to confirm final TMDb facts."""
        unknown = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates]
        if unknown:
            raise ValueError(f"unknown candidate_id: {unknown[0]}")
        if state.selected_ids is not None:
            raise ValueError("watchlist selection is already defined")
        state.selected_ids = candidate_ids
        state.confirmed = {
            candidate_id: cache.movies[candidates[candidate_id]]
            for candidate_id in candidate_ids
        }
        return json.dumps(
            {
                "movies": [
                    {
                        "candidate_id": candidate_id,
                        **fact.model_dump(mode="json"),
                    }
                    for candidate_id, fact in state.confirmed.items()
                ]
            },
            ensure_ascii=False,
        )

    return get_tmdb_details
