---
name: feature-implementation-workflow
description: Coordinate Codex research, planning, explicit user approval, failing-test authoring, implementation, and verification for new or modified software features. Use in the main or orchestrator agent before changing feature code. Do not use for read-only explanations, audits, reviews, or diagnosis without requested implementation.
---

# Feature Implementation Workflow

Use this workflow before changing code for feature work. Keep research, planning, approval, failing tests, and implementation as separate phases.

The main agent is the orchestrator. It owns user communication, preserves each phase result, reviews delegated work, and decides when new information requires returning to an earlier phase.

Use the matching project-scoped agents from `.codex/agents/` when they are available. If a named agent is unavailable, perform that phase directly and tell the user which delegation was unavailable.

Delegated agents do not ask the user directly. When an agent needs product judgment, clarification, or approval, it returns the unclear point, why it matters, reasonable options, and its recommendation. The orchestrator relays the question and routes the answer to the appropriate phase.

## Research

Run `feature-research` with the original request and relevant workspace context. It inspects the implementation, tests, configuration, documentation, data flow, and existing patterns without editing files.

Preserve its findings about relevant files, current behavior, ownership boundaries, risks, constraints, likely test surfaces, ambiguities, assumptions, and simpler alternatives.

## Planning

Run `feature-planning` with the original request and complete research result. It creates a concise plan without editing files.

Review the plan before presenting it. Reject speculative features, broad refactors, unnecessary abstractions, and changes outside the request. The plan must define:

- Observable success criteria.
- The smallest viable approach.
- Expected file and behavior changes.
- Failing tests or checks to write before implementation.
- Focused verification commands.
- Migration or rollout notes when applicable.
- Open questions, assumptions, and meaningful tradeoffs.

## Approval

Present the concrete plan to the user before editing feature code. Do not proceed until the user explicitly approves it or clearly authorizes an adjusted version.

Do not request duplicate approval when the same concrete plan has already been approved in the current conversation. Return to research or planning when feedback materially changes the technical context or approach. Ask for clarification when approval is ambiguous.

## Failing Tests

After approval, run `feature-failing-tests` with the request, research result, approved plan, and user adjustments.

The agent writes only the tests or checks and required fixtures from the approved plan, then runs the focused command to confirm the expected failure. Preserve the changed files, exact command, failure result, and test-design concerns as implementation input.

If a meaningful automated test is not viable, require the agent to explain why and propose an observable failing check or manual acceptance procedure. Do not invent low-value tests merely to satisfy the phase.

## Implementation

Run `feature-implementation` with the request, research result, approved plan, user adjustments, and failing-test result.

The implementation agent must make the minimum production change needed for the success criteria. Tests written by `feature-failing-tests` are owned input and must not be changed by the implementation agent. If one of those tests is invalid, the implementation agent reports the problem to the orchestrator, which decides whether to return it to the test agent.

Keep changes within the approved scope, preserve concurrent and user changes, follow project patterns and the code-comment policy, and remove only unused code introduced by this feature.

## Verification And Report

Review the implementation result and run any necessary focused final checks. Debug incorrect implementation before weakening tests. If new findings materially invalidate the approved plan, stop and obtain approval for the revised plan.

The final response must summarize the implemented behavior, changed areas, checks and results, unresolved issues, skipped verification, and remaining risks. State explicitly when none remain.
