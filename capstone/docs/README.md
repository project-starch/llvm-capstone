# Capstone project documentation

Everything durable the project knows about itself: architecture, the issue registry, the test
matrix, root-cause trails, and the plans in flight. Written for human developers and for any AI
coding assistant.

> **Renamed 2026-09-04.** This directory was `capstone/agent-handoff/` until it stopped being a
> handoff channel and became the documentation. `$CAPSTONE_HANDOFF_DIR` still resolves, as an
> alias for `$CAPSTONE_DOCS_DIR`.

| Path | Default |
|------|---------|
| `$CAPSTONE_DOCS_DIR` | `capstone/docs` |
| `$CAPSTONE_TMP_ROOT` (scratch) | `/tmp/capstone` |
| Shared env file | `capstone/tests/capstone-test-env.sh` |

**New here?** Start with `ONBOARDING.md` (it has a short callout for non-Claude coding agents
like Codex/Cursor, which do not auto-read `CLAUDE.md`).

## Minimal startup reading set

For a normal fresh session, read only these files first:

1. `README.md`
2. `state/current-state.md`
3. `state/current-next-step.md`

Everything else should be loaded only when the task needs it.

## Who works where (lanes)

UPDATED 2026-08-18: **there is no lane B.** The two-lane coordination set (`COORDINATION.md`, `MULTI-AGENT-WORKFLOW.md`, `AGENT-B-SETUP.md`, and lane B's own state files) is archived under `history/18-08-2026_ARCHIVED_*`. Anything below describing a second lane is historical.

Work is split across peer agent lanes plus, when needed, an external collaborator. **The
lane structure below is historical in its two-lane form** — see the 2026-08-18 note above. As of
2026-09-04 the live lanes are: this one (compiler/board/docs), an RTL lane, a synthesis lane on a
separate machine, and a paper lane. Read the roles, not the lane count:

- **Lane A** → commits to `dev` (called `capstone-bootstrap` until 2026-09-04).
  **Lane B** → committed to `capstone-bootstrap-b`, now `lane-b/capstone-bootstrap-b`. Both branch off the shared mainline; sync via
  `git merge origin/dev`. The A↔B split, hand-off rules, and the permanent
  repository rules are in **`CLAUDE.md`**; subagent roster and rules in
  **`ref/SUBAGENTS.md`** (read it before delegating). The old A↔B peer-lane guide is
  archived at `history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md`.
- **External collaborators** using their own coding agent get a **self-contained,
  stock-toolchain** task doc under `plans/` (e.g. `plans/archived/xlang-repro-task.md`, the
  cross-language reproduction task) that does *not* depend on our in-flux compiler/ABI. The
  ONBOARDING callout covers pasting `CLAUDE.md` as context for non-Claude agents.

## Directory layout

**319 markdown files.** Counts are why this index exists: the two `state/` files went two
weeks stale without anyone noticing, because nothing mapped the tree.

| directory | files | what belongs here | how to read it |
|---|---|---|---|
| `state/` | 3 | what is true **right now** | the first thing a new session reads. If it disagrees with `ref/ISSUES.md`, ISSUES.md wins. |
| `ref/` | 31 | durable quick-reference that rarely changes | `ISSUES.md` is the registry of every known defect and is the authoritative status for all of them. `SUBAGENTS.md` before delegating. |
| `design/` | 32 | architecture and **design decisions only** | a bug-fix, root-cause trail or audit is *not* a design decision — those go to `history/`. |
| `plans/` | 24 live, 21 archived | work in flight | check the status line, then check `plans/archived/README.md` — a plan's own status line does not know it was archived. |
| `history/` | 205 | dated investigation notes, root-cause trails, superseded coordination docs | append-only. Do not retro-edit a finding; add a dated correction under it. |
| `patches/` | — | out-of-tree patches | |

### The files worth knowing by name

- **`ref/ISSUES.md`** — every defect, its status and its evidence. The single most load-bearing
  document in the repo. Silicon defects are `S-nn`, compiler defects `C-nn`, QEMU `Q-nn`,
  RTL-vs-spec `R-nn`.
- **`ref/testing-matrix.md`** — what is expected to pass where.
- **`ref/HOW-TO-LAUNCH-ON-FPGA.md`** — the long form behind the `board-run` skill.
- **`ref/fpga-silicon-measurements-for-paper.md`** — where measured numbers land. Results can be
  recorded here without touching the paper, which is deliberate.
- **`ONBOARDING.md`** — fast-track setup, including the callout for non-Claude agents that do not
  auto-read `CLAUDE.md`.

### Related trees outside this directory

- `capstone/tests/fpga-repros/` — one folder per silicon defect, each a **self-contained report**
  that may already be a live link held by the hardware side. See its own `README.md`. These are
  evidence and are never pruned.
- `.claude/skills/` — procedures that auto-load when a task matches. `board-run` and `rtl-sim`.
- `CLAUDE.md` (repo root) — the permanent rules. Read it before the docs, not after.

## Current verified baseline

> **Scope note added 2026-09-04.** This list is the **QEMU/runtime** baseline and it is still
> accurate, but it accumulated before any of the silicon work and says nothing about it. For
> what is verified **on the board** — the resident bitstream, S-06/S-07/S-08/S-12, and SQLite's
> logic tests running in a capability domain — read `state/current-state.md`. For the status of
> any individual defect, `ref/ISSUES.md` outranks both.

- working sample-domain build + runtime validation
- working OpenSBI/runtime path via `capstone/caplifive-buildroot/build/local.mk`
- validated shared-region runtime proof
- validated HostCall stdout, filewrite, fileread proofs
- validated HostCall file open/close handle-lifecycle proof
- validated HostCall handle-based FILE_WRITE, FILE_READ, FILE_SYNC, FILE_STAT_BASIC, FILE_TRUNCATE proofs
- validated HostCall SQLite-facing PATH_ACCESS and PATH_DELETE proofs
- validated combined reusable file-object proof
- working baseline and split `null_blk` regressions
- validated CoreMark profile-run on Capstone PureCap ("Correct operation validated.")
  using compiled C `domain_main` rather than `coremark_domain_entry.S`
- 78 validated BEEBS benchmarks on the split host/domain runtime path; the
  newest are `matmult-float` and `whetstone` (added `atan` to the shared libm),
  completing the soft-float/libm-only FP class
- aggregate regression wrappers are available for HostCall proofs, `null_blk`,
  and the full validated BEEBS set; BEEBS is serial by default and supports
  opt-in isolated parallel runs with `RUN_ALL_BEEBS_JOBS=N`

See `state/current-state.md` for the canonical snapshot.

## Contributing rules

- treat these as local workflow overlays on top of normal LLVM/Buildroot/Linux/QEMU conventions,
- do not mark a step complete until it has been tested at the affected layer,
- keep non-trivial code documented with concise comments, especially around state transitions and ownership rules,
- after a coherent validated change set, prefer a multi-line commit message with a short subject plus a detailed body,
- for capstone-local commits, do not add a redundant `capstone` prefix to the commit subject unless the broader monorepo context requires it,
- keep manager-facing summaries as local artifacts under `$CAPSTONE_TMP_ROOT/`, not as committed repository files,
- active plans go in `plans/` (committed here); do not store project plans outside this repository.

## Read on demand

Use these only when the task actually needs them:

- **`ref/RATE-RULE.md` — why a single wedge is NOT a result on silicon, with the measured k/n.
  Read before recording, citing or acting on any board outcome.** Rescued 2026-08-18 from deep
  inside `SILICON-BLOCKER.md`, where it silently invalidated most of that document.
- `ref/known-good-controls.md` — **currently STALE** (rows last verified three bitstreams ago) and
  a preflight gate depends on it; re-verify a row before relying on it.
- `ref/SILICON-BLOCKER.md` — **SUPERSEDED**, the 2026-08-01..06 investigation. The defect it
  chased is S-06, fixed in silicon. Kept because its line numbers are cited from live repro
  folders; do not renumber or trim it.
- **`ref/ISSUES.md` — the open-issues registry (RTL/FPGA + compiler), each with a runnable repro. Read before re-investigating anything; update whenever an issue is found, characterised or closed.**
- `ref/HOW-TO-MEASURE-OVERHEAD.md` — **how overhead is measured** (bare-metal baseline, gates, traps). Read before producing or citing a ratio.
- `ref/testing-matrix.md` — compact map of test layers and entry points
- `ref/capstone-agent-test-instructions.md` — practical command cookbook
- `ref/capstone-coding-conventions.md` — local coding conventions
- `ref/delegation-guidance.md` — bounded executor rules for split agent work
- `ref/beebs-benchmark-bringup-manual.md` — exact workflow for adding one or
  more BEEBS benchmark wrappers
- `ref/capstone-purecap-pointer-model.md` — pointer/capability authority model
- `ref/project-structure-overview.md` — workspace guide
- `ref/runtime-terms-glossary.md` — terminology reference
- `design/sqlite-minimal-vfs-path.md` — concrete SQLite-facing next step and minimal VFS mapping
- `design/hostcall-file-service-v0-wire-spec.md` — wire-format and state-machine spec for the HostCall file service
- `design/stable-file-service-subset.md` — reusable HostCall file-service proposal
- `design/split-host-enclave-strategy.md` — source-backed architectural detail
- `design/hosted-libc-os-analysis.md` — hosted Linux blockers and sysroot mismatch analysis
- `design/research-decisions-log.md` — paper-worthy implementation decisions and tradeoffs, cited by commit hash
- `plans/backend-compiler-fixes.md` — known backend bugs and workarounds (from CoreMark bring-up)
- `history/README.md` — historical index and note selection guide

## History rules

- write history notes in English,
- use `DD-MM-YYYY_HH-MM-SS` in filenames,
- avoid proper names or direct references to specific people in filenames/titles,
- keep durable current guidance in `state/`, `ref/`, or `design/`, not in `history/`,
- if two history notes become near-duplicates, keep one full primary source and reduce the other to a short pointer.

## Maintenance rule

If the validated baseline or recommended workflow changes, update at least:

- `README.md`
- `new-chat-prompt.md`
- `state/current-state.md`
- `state/current-next-step.md`
- `ref/testing-matrix.md`
- `ref/capstone-agent-test-instructions.md`

Update deeper `design/` files only when their subject actually changed.

## What this does not yet mean

This does **not** yet mean that a full hosted `capstone64-unknown-linux-gnu` user-space is ready.
The current validated path is still the split host/domain runtime path.
