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

### 8. Timestamped answer / investigation history
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
- the old manual ELF-header rewrite hack is no longer needed in the default sample path.

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

If a proof file becomes stale or redundant, replace it with a shorter current proof instead of accumulating duplicate logs.

## What this does NOT yet mean

This does **not** yet imply that the whole broader hosted toolchain/runtime is ready for FFmpeg/sqlite/libpng/SPEC.

For the current recommended next milestone, see:
- `current/current-next-step.md`


