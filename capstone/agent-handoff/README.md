# Capstone agent handoff bundle

This directory stores persistent context for continuing the Capstone backend/toolchain bring-up from a new chat/session.

Location:
- `$CAPSTONE_HANDOFF_DIR` (default: `capstone/agent-handoff` inside the repository root)

Shared test environment defaults:
- `capstone/tests/capstone-test-env.sh`

Scratch log directory for future sessions:
- `$CAPSTONE_TMP_ROOT` (default: `/tmp/capstone`)

## Most important files

Current, durable working notes now live under:
- `current/`

Timestamped narrative notes / session history now live under:
- `history/`

History-note rules:
- write notes in English,
- use `DD-MM-YYYY_HH-MM-SS` in filenames,
- avoid proper names or direct references to specific people in filenames/titles,
- keep durable current-state guidance in `current/` rather than in history notes.

Agent workflow rules to preserve across sessions:
- treat the handoff rules in this directory as local workflow overlays on top of normal LLVM/Buildroot/Linux/QEMU development practices, not as replacements for subtree-native review and coding conventions,
- do not mark a step complete until it has been tested at the layer affected by that step,
- after a coherent validated change set, provide exact commit command(s) and proposed commit message(s) when a commit is appropriate, and prefer a real multi-line commit message with a short subject plus a more detailed body rather than only a one-line summary,
- document non-trivial new code with concise comments explaining protocol layouts, state transitions, and other non-obvious logic,
- keep manager-facing summary files as local artifacts (for example under `$CAPSTONE_TMP_ROOT/`) rather than committing them into the repository.

### 1. How to run tests / reproduce the validated flow
- `current/capstone-agent-test-instructions.md`

### 1a. Testing matrix / what each test layer proves
- `current/testing-matrix.md`

### 2. Current backend/compiler implementation status
- `current/capstone-backend-status-for-llm.md`

### 3. Native sample validation summary
- `current/native-sample-validation.md`

### 4. Prompt for a fresh chat
- `new-chat-prompt.md`

### 5. Current recommended next step
- `current/current-next-step.md`

### 6. Split host-enclave strategy (source-backed)
- `current/split-host-enclave-strategy.md`

### 7. Hosted libc / OS / syscall analysis
- `current/hosted-libc-os-analysis.md`

### 8. Project structure overview
- `current/project-structure-overview.md`

### 9. Coding conventions for the `capstone/` workspace layer
- `current/capstone-coding-conventions.md`

### 10. Runtime terminology glossary
- `current/runtime-terms-glossary.md`

### 11. Timestamped answer / investigation history
- `history/`

## Files intentionally not kept in this handoff bundle

The following categories were intentionally omitted to keep the handoff concise:
- long-form duplicated narrative explanations,
- intermediate exploratory logs,
- noisy build/image logs that do not add new state beyond the written summaries.

## Current verified milestone

As of the latest validated state:
- the in-tree LLVM Capstone backend builds the sample domain,
- in-tree `ld.lld` links it natively as `EM_CAPSTONE`,
- the Buildroot userspace loader accepts `EM_CAPSTONE`,
- the sample domain executes successfully in the Capstone QEMU/Buildroot runtime,
- the old manual ELF-header rewrite hack is no longer needed in the default sample path,
- `capstone/caplifive-buildroot/build/local.mk` is again present and keeps Buildroot on the local Capstone-enabled Linux/OpenSBI override path,
- rerunning `make build CAPSTONE_CC_PATH=... A=opensbi-rebuild` regenerates the OpenSBI wrapper assembly and restores the intended firmware/runtime path,
- `capstone/tests/runtime-qemu/run-shared-region-probe.sh` now passes and proves that the host-visible shared-region mutations are working again,
- `capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh` now passes and proves the first tiny split host/service request-response over shared metadata + payload,
- `capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh` now passes and proves that the same HostCall metadata ABI and borrowed payload discipline also support a second coarse host service,
- baseline `null_blk` works,
- and split `null_blk` now creates `/dev/nullb0`, completes I/O, and unloads successfully after rebuilding the package against the active kernel.

## Maintenance rule

Future sessions should keep this directory current.

If the validated baseline changes, update at least:
- `README.md`
- `current/testing-matrix.md`
- `new-chat-prompt.md`
- `current/capstone-agent-test-instructions.md`
- `current/capstone-backend-status-for-llm.md`
- `current/current-next-step.md`
- `current/split-host-enclave-strategy.md`
- `current/project-structure-overview.md`
- `current/capstone-coding-conventions.md`
- `current/runtime-terms-glossary.md`

If a proof file becomes stale or redundant, replace it with a shorter current proof instead of accumulating duplicate logs.

The obsolete draft runtime-author message that was created while the tree was still
on the wrong-firmware path should not be kept under `current/`; that state has now
been superseded by the validated notes in `history/` and `current/current-next-step.md`.

## What this does NOT yet mean

This does **not** yet imply that the whole broader hosted toolchain/runtime is ready for FFmpeg/sqlite/libpng/SPEC.

For the current recommended next milestone, see:
- `current/current-next-step.md`


