# Agent startup

Canonical project context lives in `capstone/agent-handoff/`. Do not duplicate
that information here.

For a normal fresh session, read these first:

1. `capstone/agent-handoff/README.md`
2. `capstone/agent-handoff/state/current-state.md`
3. `capstone/agent-handoff/state/current-next-step.md`

Load `capstone/agent-handoff/ref/`, `design/`, and `plans/` only when the task
needs that detail.

Before build or test commands, source:

```bash
source capstone/tests/capstone-test-env.sh
```

Rules:

- Do not vendor benchmark suites or add new benchmark submodules.
- Keep fetched benchmark sources under `$CAPSTONE_TMP_ROOT`, normally
  `/tmp/capstone`.
- Do not commit debug checkpoints, session notes, or manager-facing summaries.
- Do not add `Co-Authored-By` lines.
- Keep `capstone/agent-handoff/` current when the verified baseline or next
  milestone changes.
