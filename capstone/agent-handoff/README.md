# Capstone handoff bundle

Persistent context for continuing the Capstone work in a new session.
Works for human developers and any AI coding assistant.

| Path | Default |
|------|---------|
| `$CAPSTONE_HANDOFF_DIR` | `capstone/agent-handoff` |
| `$CAPSTONE_TMP_ROOT` (scratch) | `/tmp/capstone` |
| Shared env file | `capstone/tests/capstone-test-env.sh` |

**New here?** Start with `ONBOARDING.md`.

## Minimal startup reading set

For a normal fresh session, read only these files first:

1. `README.md`
2. `state/current-state.md`
3. `state/current-next-step.md`

Everything else should be loaded only when the task needs it.

## Directory layout

```
agent-handoff/
├── ONBOARDING.md          fast-track setup for new developers/contributors
├── new-chat-prompt.md     prompt template for resuming a session
├── codex-onboarding.md    instructions + prompt for OpenAI Codex
├── state/                 VOLATILE — rewrite after each milestone
│   ├── current-state.md   what is verified right now (short)
│   └── current-next-step.md  next concrete milestone (short)
├── ref/                   durable quick-reference (rarely changes)
│   ├── testing-matrix.md
│   ├── capstone-agent-test-instructions.md
│   ├── capstone-coding-conventions.md
│   ├── project-structure-overview.md
│   └── runtime-terms-glossary.md
├── design/                deep architecture and design docs
│   ├── sqlite-minimal-vfs-path.md
│   ├── hostcall-file-service-v0-wire-spec.md
│   ├── stable-file-service-subset.md
│   ├── split-host-enclave-strategy.md
│   ├── hosted-libc-os-analysis.md
│   └── native-sample-validation.md
├── plans/                 active WIP plans (committed, portable)
│   └── backend-compiler-fixes.md
└── history/               timestamped archival notes
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

- `ref/testing-matrix.md` — compact map of test layers and entry points
- `ref/capstone-agent-test-instructions.md` — practical command cookbook
- `ref/capstone-coding-conventions.md` — local coding conventions
- `ref/project-structure-overview.md` — workspace guide
- `ref/runtime-terms-glossary.md` — terminology reference
- `design/sqlite-minimal-vfs-path.md` — concrete SQLite-facing next step and minimal VFS mapping
- `design/hostcall-file-service-v0-wire-spec.md` — wire-format and state-machine spec for the HostCall file service
- `design/stable-file-service-subset.md` — reusable HostCall file-service proposal
- `design/split-host-enclave-strategy.md` — source-backed architectural detail
- `design/hosted-libc-os-analysis.md` — hosted Linux blockers and sysroot mismatch analysis
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
