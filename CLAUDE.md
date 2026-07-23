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

## Context & compaction

Board-debug threads here run long, so manage context deliberately. Lightly (not
every turn) assess whether it's a good moment to `/compact`, and **proactively
recommend it in one line** when work hits a natural checkpoint AND the important
state is already captured in committed docs/memory so it can be safely summarized.
Do **not** recommend it mid-task, during active debugging, or while un-captured
details still matter. Briefly say why the timing is safe (or why to postpone).
Never compact unilaterally — you can only recommend; it is the user's call.

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

## Delegating to subagents

Delegation is **opt-in and at the main session's discretion** — not automatic.
The main (Opus) session owns planning, synthesis, reviewing subagent output, and
all final decisions. Subagents start cold, so any that could write or run tests
**inherit this file's hard constraints** (no real-person names; serialize QEMU
suites — the shared `rootfs.ext2` write-lock means never two in parallel;
`ninja -j90` never `-j112`; never commit unless asked; submodule source stays
uncommitted). Subagents do not recurse.

- **Delegate** (notify the user when it's substantial): broad read-only code/file
  search → the built-in **Explore** agent; running the regression corpus / lit /
  QEMU suites for validation → the **corpus-runner** agent (Sonnet, read-only,
  serialized, never touches the board); bounded multi-step research with a clear
  question → **general-purpose**.
- **Keep in the main Opus session** (never delegate): all FPGA/board sessions
  (board etiquette + secret token + can't be parallelized); compiler/codegen and
  capability-ABI changes; subtle-correctness or concurrency debugging; the paper;
  commits; and anything involving real-person names.
