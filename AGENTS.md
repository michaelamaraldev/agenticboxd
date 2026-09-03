# Agent Instructions

## Feature Work

When you are the main or orchestrator agent and the user asks to create a new feature or modify an existing feature, load and follow the `feature-implementation-workflow` skill before making code changes. That skill is the authoritative workflow for feature research, planning, user approval, implementation, and verification.

## General Rules

### Think Before Coding

- State assumptions, uncertainty, and meaningful tradeoffs explicitly.
- If a request has multiple reasonable interpretations, present them or ask before choosing.
- Push back when a simpler approach would satisfy the goal.
- Stop when confused. Name what is unclear and ask for clarification.

### Keep Changes Simple And Surgical

- Use the minimum code that solves the approved goal.
- Touch only files needed for the request.
- Match the existing style and project patterns.
- Do not add speculative features, abstractions, configurability, or unrelated cleanup.
- Preserve user changes. Do not revert work you did not make.
- Remove only unused code created by your own change.

### No Comments In Code

- Do not write comments in code unless I explicitly call you to write then. Do not explain decisions, justify numbers, narrate bug fixes, or describe how something was tuned.
- Exactly two exceptions are allowed:
  - Section markers inside templates or configuration blocks, such as `<!-- combo -->`.
  - One-line docstrings on functions, classes, or types, such as `/** Faz X. */`.
- If an explanation does not fit in one line, put it in the final response instead of a code comment.

### Verify And Report

- Define success criteria for non-trivial work.
- Run focused checks that match the risk and scope.
- If verification fails, debug the implementation before weakening checks.
- Final responses must summarize what changed, how it was verified, and any unresolved issues.

## Boundaries

### Always
- Load `langgraph-agent-patterns` and `langgraph-state-management` before structuring
  any node or state schema.
- Keep the watchlist as the only source of recommendable candidates.

### Ask First
- Adding a new agent/node beyond the three described in the plan.
- Any change to how facts are validated against TMDb.

### Never
- git init, commits, README.md, CHANGELOG.md, or any documentation artifact.
- Comments in code beyond the two documented exceptions.
- Recommending anything outside the user's watchlist.