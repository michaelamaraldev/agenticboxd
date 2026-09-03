from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    CineResult,
    HistoryContext,
    RecommendationChoices,
    TasteProfile,
    TmdbCache,
    WatchlistSelection,
)


def test_structured_contracts_are_pydantic_models() -> None:
    assert TasteProfile(likes=("natural dialogue",), dislikes=()).likes
    assert RecommendationChoices(recommendations=()).recommendations == ()
    assert CineResult().recommendations == ()
    assert TmdbCache().movies == {}
    assert HistoryContext(relevant_patterns=("pattern",)).relevant_patterns


@pytest.mark.parametrize(
    "candidate_ids",
    [
        (1, 1),
        tuple(range(1, 12)),
    ],
)
def test_watchlist_selection_rejects_duplicates_and_more_than_ten(
    candidate_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError):
        WatchlistSelection(candidate_ids=candidate_ids)
