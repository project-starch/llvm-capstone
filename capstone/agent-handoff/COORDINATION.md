# COORDINATION — live ownership between the two agents

Single source of truth for **who owns what right now**. Update this file
*before* you start on a subsystem/submodule and *when you hand off*. Read it at
the start of every session. Rules behind it: `MULTI-AGENT-WORKFLOW.md`.
Setup for the second agent: `AGENT-B-SETUP.md`.

- **Agent-A** — clone `/home/alexey/dev/llvm-capstone`, branch `capstone-bootstrap`.
- **Agent-B** — clone `/home/alexey/dev/llvm-capstone-b`, branch `capstone-bootstrap-b`.

---

## Active ownership  _(edit before you start / on handoff)_

| Owner | Subsystem | Submodules / dirs held | Branch |
|---|---|---|---|
| Agent-A | (current: SQLite Stage-2 corpus, firmware/monitor) | `caplifive-buildroot` → `opensbi`, `capstone-sbi-domain/capstone-sbi`; `capstone/capstone-c`; `capstone/benchmarks/sqlite`; `capstone/tests` | `capstone-bootstrap` |
| Agent-B | unassigned — awaiting task (clone `/home/alexey/dev/llvm-capstone-b`) | none yet | `capstone-bootstrap-b` |

Suggested non-overlapping split (pick when B's task is set):
- Compiler/codegen (`llvm/`, in-tree) + `capstone/capstone-qemu` — one owner.
- Firmware/monitor (nested `caplifive-buildroot`/`opensbi`/`capstone-sbi`) + `capstone-c` — the other.
- `capstone/paper` — whoever is writing; low build-contention, coordinate edits here.

## Claimed / do-not-touch  _(hold list)_

| Path or submodule | Held by | Until |
|---|---|---|
| `capstone/agent-handoff/state/current-state.md` + `current-next-step.md` | Agent-A (base files) | ongoing — Agent-B uses `*.B.md` |
| _add entries as you claim exclusive edit rights_ | | |

## Submodule-bump log  _(append after every gitlink change — this is where conflicts hide)_

| Date | Agent | Submodule | Old→New SHA | Superproject commit |
|---|---|---|---|---|
| 2026-07-08 | Agent-A | (main repo) added `capstone/paper` @ db9142f (opt-in, update=none) | — → d4959767 | d4959767d27c |

## Sync log  _(append at each integration point)_

| Date | Action | Detail |
|---|---|---|
| 2026-07-08 | scaffolding | Added `MULTI-AGENT-WORKFLOW.md`, `AGENT-B-SETUP.md`, this file; paper submodule committed (`d4959767`). Push required for Agent-B's clone to pull them. |
| 2026-07-08 | Agent-B online | Agent-B came online in clone `/home/alexey/dev/llvm-capstone-b` on `capstone-bootstrap-b`. Verified isolation: `CAPSTONE_REPO_ROOT`/`CAPSTONE_CLANG` inside B clone, `CAPSTONE_TMP_ROOT=/tmp/capstone-b`, `CLAUDE_CONFIG_DIR=~/.claude-b` (own creds), remote=project-starch/llvm-capstone, clean tree. Created `state/current-{state,next-step}.B.md`. LLVM build + buildroot/rootfs not yet built (fresh clone). Awaiting task assignment; no submodule owned. |

---

### Handoff etiquette
- Before bumping a submodule: check the **Claimed** list; if someone holds it, wait or coordinate here.
- After bumping: append to the **Submodule-bump log** in the same commit.
- Keep the branches close — merge B→A at checkpoints rather than letting gitlink SHAs diverge for weeks.
- Durable facts both agents need go in a committed `agent-handoff/` doc, **not** only in an agent's private memory (memory doesn't cross clones/accounts).
