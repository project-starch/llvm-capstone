# Capstone LLVM fork — Claude Code workspace

## Read first

Set up the environment, then read the minimal handoff set:

```bash
source capstone/tests/capstone-test-env.sh
```

- `capstone/agent-handoff/README.md`
- `capstone/agent-handoff/state/current-state.md`
- `capstone/agent-handoff/state/current-next-step.md`

New to the project? See `capstone/agent-handoff/ONBOARDING.md`.

## Hard constraints

- **Never mention any real person by name — anywhere.** No PI, supervisor,
  colleague, board owner, or collaborator names in commits, code, docs, reports,
  or any committed/shared content. Use neutral roles ("the board owner", "the
  collaborator", "the PI"). This is permanent and absolute. (Upstream `lldb/`,
  `llvm/` etc. files are not ours — leave their names alone.)
- No `Co-Authored-By:` lines in commits.
- Never commit debug/report files (`*_DEBUG_CHECKPOINT.md`, session notes).
- Active plans live in `capstone/agent-handoff/plans/` (committed, portable across machines and agents).
- Manager-facing summaries go under `/tmp/capstone/`, not into the repo.

## Where things live

| What | Where |
|------|-------|
| Current state + next step | `capstone/agent-handoff/state/` |
| Test matrix + cookbook | `capstone/agent-handoff/ref/` |
| Architecture + design docs | `capstone/agent-handoff/design/` |
| Active WIP plans | `capstone/agent-handoff/plans/` |
| Bug-fix investigations, root-cause trails, audits | `capstone/agent-handoff/history/` (dated `DD-MM-YYYY_HH-MM-SS_name.md`) |
| Archived session notes | `capstone/agent-handoff/history/` |

**`design/` is for design decisions and architecture only.** A bug fix,
root-cause investigation, or audit — even a substantial one — is *not* a design
decision; it belongs in `history/` as a dated note, not in `design/`.
