from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from ..models import HistoryContext, LetterboxdData


def history_prompt() -> str:
    """Build the viewing-history agent prompt."""
    return (
        "Relate the user's current request to the complete Letterboxd diary in its original "
        "order. Identify between 1 and 5 useful viewing patterns and up to 3 rewatch signals. "
        "Use dates, ratings, tags, and rewatch markers only as evidence. Do not recommend, "
        "reject, rank, or filter any movie. Do not invent viewing history. Return the structured "
        "response immediately without intermediate analysis or prose."
    )


def history_request(request: str, data: LetterboxdData) -> str:
    return f"REQUEST:\n{request}\nDIARY_TSV:\n{data.diary_tsv}"


def create_history_agent(model: BaseChatModel) -> Runnable[Any, Any]:
    return create_agent(
        model,
        tools=[],
        system_prompt=history_prompt(),
        response_format=ToolStrategy(HistoryContext, handle_errors=False),
        name="history_agent",
    )
