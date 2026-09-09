from __future__ import annotations

import json
from dataclasses import dataclass

from langchain.tools import tool

from ..agents.history import create_history_agent, history_message
from ..local_data import create_history_search_tool
from ..models import HistoryContext


@dataclass(slots=True)
class HistoryToolState:
    context: HistoryContext | None = None
    calls: int = 0


def create_analyze_history_tool(
    model, evidence_store, request, state, budget, personal_evidence
):
    @tool
    def analyze_history(question: str = "") -> str:
        """Investigate relevant diary periods or rewatch signals for the current request."""
        state.calls += 1
        store = evidence_store.fork()
        scope = budget.scope("history", model_limit=4, tool_limit=4)
        agent = create_history_agent(model, [create_history_search_tool(store)], [scope])
        result = agent.invoke({"messages": [history_message(request, store.summary(("diary",)), question)]},
                              config={"max_concurrency": 1, "recursion_limit": 20})
        context = result.get("structured_response")
        if not isinstance(context, HistoryContext):
            raise TypeError("invalid history agent structured response")
        refs = {ref for signal in (*context.relevant_patterns, *context.rewatch_signals) for ref in signal.evidence_ids}
        store.validate_refs(refs)
        evidence = {ref: store.by_id[ref] for ref in refs}
        personal_evidence.update(evidence)
        state.context = context
        return json.dumps(
            {
                "context": context.model_dump(mode="json"),
                "evidence": evidence,
                "required_final_personal_evidence_ids": sorted(refs),
                "final_instruction": (
                    "Every recommendation must copy at least one diary ID from "
                    "required_final_personal_evidence_ids."
                ),
            },
            ensure_ascii=False,
        )
    return analyze_history
