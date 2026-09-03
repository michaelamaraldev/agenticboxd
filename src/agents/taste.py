from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from ..models import LetterboxdData, TasteProfile


def taste_prompt() -> str:
    """Build the taste agent prompt."""
    return (
        "Analyze all supplied Letterboxd ratings and reviews. "
        "Infer stable likes and dislikes only from that evidence. "
        "Prefer recurring patterns over isolated entries and do not invent preferences. "
        "Return the structured response immediately without intermediate analysis or prose."
    )


def taste_request(data: LetterboxdData) -> str:
    return f"RATINGS_TSV:\n{data.ratings_tsv}\nREVIEWS_TSV:\n{data.reviews_tsv}"


def create_taste_agent(model: BaseChatModel) -> Runnable[Any, Any]:
    return create_agent(
        model,
        tools=[],
        system_prompt=taste_prompt(),
        response_format=ToolStrategy(TasteProfile, handle_errors=False),
        name="taste_agent",
    )
