from __future__ import annotations

import json
from dataclasses import dataclass

from langchain.tools import tool

from ..agents.taste import create_taste_agent, taste_message
from ..local_data import EvidenceStore, create_taste_search_tool
from ..models import TasteProfile


@dataclass(slots=True)
class TasteToolState:
    profile: TasteProfile | None = None
    calls: int = 0


def profile_refs(profile):
    return {ref for signal in (*profile.likes, *profile.dislikes) for ref in signal.evidence_ids}


def run_taste(model, evidence_store: EvidenceStore, budget, question=""):
    store = evidence_store.fork()
    scope = budget.scope("taste", model_limit=4, tool_limit=4)
    agent = create_taste_agent(model, [create_taste_search_tool(store)], [scope])
    summary = store.summary(("ratings", "reviews", "likes", "favorites", "rankings"))
    result = agent.invoke({"messages": [taste_message(summary, question)]}, config={"max_concurrency": 1, "recursion_limit": 20})
    profile = result.get("structured_response")
    if not isinstance(profile, TasteProfile):
        raise TypeError("invalid taste agent structured response")
    refs = profile_refs(profile)
    store.validate_refs(refs)
    return profile, {ref: store.by_id[ref] for ref in refs}


def create_analyze_taste_tool(model, evidence_store, state, budget, personal_evidence):
    @tool
    def analyze_taste(question: str = "") -> str:
        """Ask the taste specialist to investigate additional personal evidence relevant to a question."""
        state.calls += 1
        profile, evidence = run_taste(model, evidence_store, budget, question)
        state.profile = profile
        personal_evidence.update(evidence)
        return json.dumps({"profile": profile.model_dump(mode="json"), "evidence": evidence}, ensure_ascii=False)
    return analyze_taste
