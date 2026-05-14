# Capstone agent handoff bundle

This directory stores persistent context for continuing the Capstone work in a new
chat/session.

Location:
- `$CAPSTONE_HANDOFF_DIR` (default: `capstone/agent-handoff`)

Shared environment defaults:
- `capstone/tests/capstone-test-env.sh`

Scratch logs:
- `$CAPSTONE_TMP_ROOT` (default: `/tmp/capstone`)

## Minimal startup reading set

For a normal fresh session, read only these files first:

1. `README.md`
2. `current/current-state.md`
3. `current/current-next-step.md`

Everything else should be loaded only when the task needs it.

## Directory layout

- `current/` — compact durable current-state notes
- `history/` — timestamped archival notes and chronology
- `new-chat-prompt.md` — compact prompt template for a new chat

## Current verified baseline

At a high level, the repository currently has:

- working sample-domain build + runtime validation,
- working restored OpenSBI/runtime path via `capstone/caplifive-buildroot/build/local.mk`,
- validated shared-region runtime proof,
- validated HostCall stdout proof,
- validated HostCall filewrite proof,
- validated HostCall fileread reverse-direction proof,
- working baseline and split `null_blk` regressions.

See `current/current-state.md` for the concise canonical state snapshot.

## Workflow rules to preserve

- treat these handoff rules as local workflow overlays on top of normal LLVM/Buildroot/Linux/QEMU conventions,
- do not mark a step complete until it has been tested at the affected layer,
- keep non-trivial code documented with concise comments, especially around state transitions and ownership rules,
- after a coherent validated change set, provide exact commit command(s) and prefer a multi-line commit message with a short subject plus a detailed body,
- keep manager-facing summaries as local artifacts under `$CAPSTONE_TMP_ROOT/`, not as committed repository files.

## Read on demand

Use these only when the task actually needs them:

- `current/testing-matrix.md` — compact map of test layers and entry points
- `current/capstone-agent-test-instructions.md` — practical command cookbook
- `current/stable-file-service-subset.md` — first reusable HostCall file-service proposal
- `current/split-host-enclave-strategy.md` — source-backed architectural detail
- `current/hosted-libc-os-analysis.md` — hosted Linux blockers and sysroot mismatch analysis
- `current/capstone-backend-status-for-llm.md` — backend/compiler implementation detail
- `current/native-sample-validation.md` — sample-domain validation detail
- `current/project-structure-overview.md` — workspace guide
- `current/capstone-coding-conventions.md` — local coding conventions
- `current/runtime-terms-glossary.md` — terminology reference
- `history/README.md` — historical index and note selection guide

## History rules

- write history notes in English,
- use `DD-MM-YYYY_HH-MM-SS` in filenames,
- avoid proper names or direct references to specific people in filenames/titles,
- keep durable current guidance in `current/`, not in `history/`,
- if two history notes become near-duplicates, keep one full primary source and reduce the other to a short pointer.

## Maintenance rule

If the validated baseline or recommended workflow changes, update at least:

- `README.md`
- `new-chat-prompt.md`
- `current/current-state.md`
- `current/current-next-step.md`
- `current/testing-matrix.md`
- `current/capstone-agent-test-instructions.md`

Update deeper reference files only when their subject actually changed.

## What this does not yet mean

This does **not** yet mean that a full hosted `capstone64-unknown-linux-gnu` user-space is ready.
The current validated path is still the split host/domain runtime path.


