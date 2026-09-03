from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain.tools import tool
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from agents.taste import taste_request
from models import LetterboxdData, TasteProfile


@dataclass(slots=True)
class TasteToolState:
    profile: TasteProfile | None = None
    calls: int = 0


def create_analyze_taste_tool(
    taste_agent: Runnable[Any, Any], data: LetterboxdData, state: TasteToolState
) -> BaseTool:
    """Build the taste-analysis tool used by the recommendation agent."""

    @tool
    def analyze_taste() -> str:
        """Ask the taste agent to infer preferences from all Letterboxd ratings and reviews."""
        state.calls += 1
        # O perfil de gosto é calculado uma vez e reaproveitado na sessão.
        if state.profile is None:
            result = taste_agent.invoke(
                {"messages": [{"role": "user", "content": taste_request(data)}]}
            )
            if not isinstance(result, Mapping):
                raise ValueError("invalid taste agent result")
            profile = result.get("structured_response")
            if not isinstance(profile, TasteProfile):
                raise ValueError("invalid taste agent structured response")
            state.profile = profile
        return state.profile.model_dump_json()

    return analyze_taste
