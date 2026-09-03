from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel


def selection_prompt() -> str:
    """Build the candidate-selection prompt."""
    return (
        "Interpret the user's request freely, including cinematic references from your general "
        "knowledge. Treat the request as primary and the taste profile as personalization and a "
        "tiebreaker. Consider the entire verified watchlist catalog, choose the strongest matching "
        "candidates, and rank them before responding. Return between 1 and 10 unique "
        "candidate_ids. Never return every match or select an ID outside the catalog. Return the "
        "structured response immediately without intermediate analysis or prose."
    )


def selection_request(utterance: str, taste_profile: str, catalog_tsv: str) -> str:
    return (
        f"UTTERANCE:\n{utterance}\nTASTE_PROFILE_JSON:\n{taste_profile}"
        f"\nVERIFIED_WATCHLIST_TSV:\n{catalog_tsv}END_VERIFIED_WATCHLIST\n"
        "FINAL_INSTRUCTION:\nReturn no more than 10 candidate_ids. If more movies match, "
        "rank them and keep only the 10 strongest. Never list every matching movie."
    )


def final_prompt() -> str:
    """Build the final-recommendation prompt."""
    return (
        "Choose up to 3 of the supplied TMDb-confirmed candidates that best satisfy the user's "
        "request. Rank by fit and use diversity only as a tiebreaker. Write concise fit_reason "
        "text in Brazilian Portuguese. You may use general film knowledge to interpret the "
        "request, but make factual claims about candidates only from the supplied TMDb data. Copy "
        "selection_position exactly and do not recommend any other movie. Return the structured "
        "response immediately without intermediate analysis or prose."
    )


def final_request(utterance: str, taste_profile: str, details_json: str) -> str:
    return (
        f"UTTERANCE:\n{utterance}\nTASTE_PROFILE_JSON:\n{taste_profile}"
        f"\nSELECTED_TMDB_FACTS_JSON:\n{details_json}"
    )


def create_recommendation_agent(
    model: BaseChatModel,
    system_prompt: str,
    response_model: type[BaseModel],
) -> Runnable[Any, Any]:
    return create_agent(
        model,
        tools=[],
        system_prompt=system_prompt,
        response_format=ToolStrategy(response_model, handle_errors=False),
        name="recommendation_agent",
    )
