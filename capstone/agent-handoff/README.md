# Capstone handoff bundle

Persistent context for continuing the Capstone work in a new session.
Works for human developers and any AI coding assistant.

| Path | Default |
|------|---------|
| `$CAPSTONE_HANDOFF_DIR` | `capstone/agent-handoff` |
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

Work is split across two peer agent lanes plus, when needed, an external collaborator:

- **Lane A** → commits to `capstone-bootstrap`. **Lane B** → commits to
  `capstone-bootstrap-b`. Both branch off the shared mainline; sync via
  `git merge origin/capstone-bootstrap`. The A↔B split, hand-off rules, and the permanent
  repository rules are in **`CLAUDE.md`**; subagent roster and rules in
  **`ref/SUBAGENTS.md`** (read it before delegating). The old A↔B peer-lane guide is
  archived at `history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md`.
- **External collaborators** using their own coding agent get a **self-contained,
  stock-toolchain** task doc under `plans/` (e.g. `plans/xlang-repro-task.md`, the
  cross-language reproduction task) that does *not* depend on our in-flux compiler/ABI. The
  ONBOARDING callout covers pasting `CLAUDE.md` as context for non-Claude agents.

## Directory layout

```
agent-handoff/
├── ONBOARDING.md          fast-track setup (incl. non-Claude-agent + collaborator callout)
├── ref/SUBAGENTS.md       subagent roster, inherited rules, how to read their output
├── new-chat-prompt.md     prompt template for resuming a session
├── state/                 VOLATILE — rewrite after each milestone
│   ├── current-state.md   what is verified right now (short)
│   └── current-next-step.md  next concrete milestone (short)
├── ref/                   durable quick-reference (rarely changes)
│   ├── testing-matrix.md
│   ├── capstone-agent-test-instructions.md
│   ├── capstone-coding-conventions.md
│   ├── delegation-guidance.md
│   ├── beebs-benchmark-bringup-manual.md
│   ├── project-structure-overview.md
│   └── runtime-terms-glossary.md
├── design/                deep architecture and DESIGN DECISIONS only
│                          (bug-fix investigations/root-cause/audits -> history/)
│   ├── sqlite-minimal-vfs-path.md
│   ├── hostcall-file-service-v0-wire-spec.md
│   ├── stable-file-service-subset.md
│   ├── split-host-enclave-strategy.md
│   ├── hosted-libc-os-analysis.md
│   ├── research-decisions-log.md
│   └── native-sample-validation.md
├── plans/                 active WIP plans (committed, portable)
│   └── backend-compiler-fixes.md
└── history/               timestamped archival notes + bug-fix/root-cause trails
```

## Current verified baseline

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
- MicroPython EXTRA+MPZ executes all 917 direct tests in the upstream default base directories
  under QEMU via resumable chunks (`598 PASS / 87 FAIL / 0 FAULT / 232 UNSCORED`); all 200 direct
  optional single-interpreter files were also attempted, for 1,117 executed files total
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
