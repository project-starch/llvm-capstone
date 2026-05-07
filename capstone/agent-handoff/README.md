# Capstone agent handoff bundle

This directory stores persistent context for continuing the Capstone backend/toolchain bring-up from a new chat/session.

Location:
- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff`

Scratch log directory for future sessions:
- `/tmp/alexey`

## Most important files

### 1. How to run tests / reproduce the validated flow
- `capstone-agent-test-instructions.md`

### 1a. Testing matrix / what each test layer proves
- `testing-matrix.md`

### 2. Current backend/compiler implementation status
- `capstone-backend-status-for-llm.md`

### 3. Native sample validation summary
- `native-sample-validation.md`

### 4. Prompt for a fresh chat
- `new-chat-prompt.md`

### 5. Current recommended next step
- `current-next-step.md`

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
- `testing-matrix.md`
- `new-chat-prompt.md`
- `capstone-agent-test-instructions.md`
- `capstone-backend-status-for-llm.md`
- `current-next-step.md`

If a proof file becomes stale or redundant, replace it with a shorter current proof instead of accumulating duplicate logs.

## What this does NOT yet mean

This does **not** yet imply that the whole broader hosted toolchain/runtime is ready for FFmpeg/sqlite/libpng/SPEC.

For the current recommended next milestone, see:
- `current-next-step.md`


