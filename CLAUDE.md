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

- No `Co-Authored-By:` lines in commits.
- Never commit debug/report files (`*_DEBUG_CHECKPOINT.md`, session notes).
- Active plans live in `capstone/agent-handoff/plans/` (committed, portable across machines and agents).
- Manager-facing summaries go under `/tmp/capstone/`, not into the repo.
- **Never name people in committed files.** Do not write the PI, co-PI, any collaborator, or
  any individual's name (or personal build hostnames like `root@<name>`) into any file that
  gets committed. Refer to roles generically: "the project lead", "the collaborator", "the
  team". Names, meeting attributions, and questions/notes addressed to a person go under
  `/tmp/capstone/` only — never committed, never pushed. (Functional exceptions that are not
  name-drops: real dependency URLs in `.gitmodules` and published-paper citation URLs.)
- **Never commit or share the FPGA console link or token.** The board URL/host and its token
  are secret. Keep the full URL in an env var (`FPGA_URL`) only, for the duration of a run;
  in any committed text use the placeholder `<FPGA-CONSOLE-URL>`. Never echo it into a
  captured/committed log.

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
