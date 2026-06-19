# Delegation guidance

Use this when splitting work across multiple coding agents. It is intentionally
model-neutral: task prompts may name a specific agent, but repository guidance
should describe the role and limits.

## Bounded executor contract

Treat secondary agents as bounded executors, not autonomous architects or lead
debuggers.

Every delegated prompt should include the task-specific budget and these rules:

- The newest prompt wins over prior session summaries or previous tasks.
- Do not perform open-ended debugging.
- Use at most two attempts total:
  - Attempt 1: implement or inspect the most likely bounded change.
  - Attempt 2: make one focused correction based only on concrete evidence.
- After two failed attempts, stop immediately and return a diagnostic report.
- Do not keep trying alternative fixes beyond the attempt budget.
- Do not make architectural changes unless explicitly requested.
- Do not broaden the task scope.
- Do not modify unrelated files.
- If the task is ambiguous, risky, or requires changing core algorithms, stop
  and ask for clarification instead of guessing.
- Prefer small, reviewable patches.
- Preserve LLVM coding style and existing project conventions.

## Good delegation targets

- Focused tests and regression coverage.
- Comments and documentation updates.
- Small NFC cleanups.
- Code search and comparison with similar LLVM patterns.
- Bounded source probes with explicit stop conditions.
- Diagnostic reports with exact failures, commands, and touched files.

## Keep with the lead engineer

- Core algorithm changes.
- ABI or capability-provenance semantics.
- Pass-manager, invalidation, MemorySSA, or CaptureTracking logic.
- Difficult runtime debugging.
- Final integration and reviewer-response strategy.
- Any task that has already exceeded a bounded executor's attempt budget.

## Diagnostic report format

If a bounded executor stops without validation, require:

- Problem investigated.
- Commands run and evidence gathered.
- Changes attempted.
- Results and exact failure text.
- Assumptions.
- Files changed.
- Recommended next action for the lead engineer.
