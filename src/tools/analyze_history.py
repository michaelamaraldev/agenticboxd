from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain.tools import tool
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ..agents.history import history_request
from ..models import HistoryContext, LetterboxdData


@dataclass(slots=True)
class HistoryToolState:
    context: HistoryContext | None = None
    calls: int = 0


def create_analyze_history_tool(
    history_agent: Runnable[Any, Any], data: LetterboxdData, state: HistoryToolState
) -> BaseTool:
    """Build the tool that relates the current request to Letterboxd viewing history."""

    @tool
    def analyze_history(request: str) -> str:
        """Analyze the complete diary for patterns relevant to the current request."""
        state.calls += 1
        result = history_agent.invoke(
            {"messages": [{"role": "user", "content": history_request(request, data)}]}
        )
        if not isinstance(result, Mapping):
            raise ValueError("invalid history agent result")
        context = result.get("structured_response")
        if not isinstance(context, HistoryContext):
            raise ValueError("invalid history agent structured response")
        state.context = context
        return context.model_dump_json()

    return analyze_history
