# Multi-agent workflow — running two Claude Code agents on this repo

Two equally-capable Opus/Claude Code agents (call them **Agent-A** = the existing
`capstone-bootstrap` agent, **Agent-B** = the new one) can work this repo in
parallel *if* they stay physically and logically isolated. This repo has three
properties that make naive sharing painful — plan around them.

## Why this repo is not worktree-friendly

Measured 2026-07-08:

| Fact | Consequence |
|---|---|
| `llvm/cmake-build-debug` is **10 GB** and **gitignored** | A fresh checkout (worktree *or* clone) starts with an empty build dir. There is no way to "share" a build cheaply without symlinking one clang that both agents then fight over on rebuild. |
| **4-level nested submodule chain** (main → `caplifive-buildroot` → `opensbi` + `capstone-sbi-domain/capstone-sbi` → nested `capstone-sbi`) | `git worktree` + nested submodules is fragile; linked worktrees share submodule `.git` and need manual per-worktree `submodule update`. |
| **`rootfs.ext2` is 2 GB with a single write-lock** — never run two QEMU matrix suites against the same one | Worktrees sharing the buildroot output tree keep this contention. **Separate clones each own a `rootfs.ext2`, so the two agents can test in parallel.** |

## Recommended isolation: a second full clone

```bash
# Agent-B gets its own directory, its own submodules, its own build dirs, its own rootfs.
git clone https://github.com/project-starch/llvm-capstone /home/alexey/dev/llvm-capstone-b
cd /home/alexey/dev/llvm-capstone-b
git checkout -b capstone-bootstrap-b origin/capstone-bootstrap   # branch, see below
git submodule update --init --recursive
# then build LLVM + buildroot in this clone as normal
```

Cost: ~30 GB extra disk + one LLVM/buildroot rebuild. Buys: true test
parallelism (independent `rootfs.ext2`), no worktree/submodule fragility, fully
separate build artifacts.

**Carve-out — use a worktree instead only if Agent-B does pure docs/analysis with
no builds and no QEMU runs:**
```bash
git worktree add ../llvm-capstone-docs -b capstone-bootstrap-docs
```
That avoids duplicating 32 GB. Do **not** use a worktree for any agent that
compiles LLVM, rebuilds firmware, or runs the QEMU matrix.

## Branch strategy

- Agent-A stays on **`capstone-bootstrap`**.
- Agent-B works on **`capstone-bootstrap-b`** (or a task-descriptive name).
- **Never both commit to the same branch.** Integrate by merging B → A (or both →
  a shared integration branch) at checkpoints, via normal PR/merge, not by
  pushing to each other's branch.
- Rebase/merge deliberately at sync points; don't leave the branches diverging
  for weeks (the nested-submodule gitlink SHAs are the thing that conflicts —
  see below).

## The sharp edge: submodule-pointer (gitlink) conflicts

The most likely merge conflict is **not** in source files — it's in the gitlink
SHAs when both agents bump a submodule. Rule: **one owning agent per submodule at
a time.** Partition by subsystem so the agents rarely touch the same submodule:

| Subsystem | Submodule(s) | Suggested owner |
|---|---|---|
| Compiler / codegen | `llvm/` (in-tree, not a submodule) | whoever owns LLVM work |
| Emulator | `capstone/capstone-qemu` | keep clean; one owner |
| Monitor / firmware | `caplifive-buildroot` → `opensbi`, `capstone-sbi-domain/capstone-sbi` | firmware owner |
| Rust monitor lang | `capstone/capstone-c` | firmware owner |
| Benchmarks / probes / docs | main repo `capstone/…` (no submodule) | either |

If both must touch one submodule, coordinate through `COORDINATION.md` (below) and
merge that submodule's bump before the other agent starts.

## Shared-file hazards inside `agent-handoff/`

These are the files two agents will silently clobber:

- **`state/current-state.md`, `state/current-next-step.md`** — single-writer by
  design. Two agents editing them = confusion. **Fix:** each agent keeps its own,
  e.g. `state/current-state.A.md` / `state/current-state.B.md`, or Agent-B keeps
  its state under its own clone only. Do **not** have both write the base files.
- **`plans/`** — partition by plan file; each plan owned by one agent. Filenames
  are already descriptive, so collisions are avoidable by convention.
- **`history/DD-MM-YYYY_HH-MM-SS_*.md`** — timestamped, append-only, low collision
  risk. Safe to share; still, prefix the slug with the agent if two land in the
  same second.
- **`ref/` matrix + cookbook** — shared reference; treat as one-owner-at-a-time for
  edits, read-only for the other.

## Claude Code metadata (won't auto-share — and that's mostly fine)

- **Memory dir** (`~/.claude/projects/<path-hash>/memory/` + `MEMORY.md`) is keyed
  by the *clone's absolute path*. Two clones ⇒ two separate memory stores. Learnings
  do **not** propagate automatically. If a durable fact should be known to both,
  write it into a committed doc (`agent-handoff/…`), not only into memory.
- **Scratchpad** dirs are session-keyed and always isolated — no action needed.
- **CLAUDE.md** is checked in, so both clones read the same project instructions.

## Coordination protocol

Keep a committed, frequently-updated `capstone/agent-handoff/COORDINATION.md` that
is the single source of "who owns what right now":

```
## Active ownership (update before you start / when you hand off)
- Agent-A (capstone-bootstrap): <subsystem>, submodules {…}, files {…}
- Agent-B (capstone-bootstrap-b): <subsystem>, submodules {…}, files {…}
## Claimed / do-not-touch
- <path or submodule> — held by <agent> until <event>
## Sync log
- <date> merged B→A at <sha>; submodule bumps reconciled: {…}
```

Push/pull through `origin`; the branches (not a shared working dir) are the
coordination channel. Before bumping any submodule, check `COORDINATION.md`; after,
note it in the sync log.

## Do-not hazards (both agents)

- Never run two QEMU matrix suites against the **same** `rootfs.ext2` (the whole
  reason for separate clones).
- Keep `capstone-qemu` and `caplifive-buildroot` submodules clean.
- `git add` exact paths only; commit only when the human asks; no `Co-Authored-By:`
  lines; never commit debug/report files.
