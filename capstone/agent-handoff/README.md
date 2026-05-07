# Capstone agent handoff bundle

This directory stores persistent context for continuing the Capstone backend/toolchain bring-up from a new chat/session.

Location:
- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff`

Scratch log directory for future sessions:
- `/tmp/alexey`

## Most important files

### 1. How to run tests / reproduce the validated flow
- `capstone-agent-test-instructions.md`

### 2. Current backend/compiler implementation status
- `capstone-backend-status-for-llm.md`

### 3. Prompt for a fresh chat
- `new-chat-prompt.md`

### 4. Current recommended next step
- `current-next-step.md`

## Useful verification logs copied from the scratch log directory

These files record the latest verified native sample flow:
- `capstone-my-domain-build-native.txt`
- `capstone-my-domain-readobj-native.txt`
- `capstone-qemu-native-proof.txt`
- `capstone-lld-lit.txt`

## Files intentionally not kept in this handoff bundle

The following categories were intentionally omitted to keep the handoff concise:
- long-form duplicated narrative explanations,
- intermediate exploratory logs,
- noisy build/image logs that do not add new state beyond the kept proof files.

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
- `new-chat-prompt.md`
- `capstone-agent-test-instructions.md`
- `capstone-backend-status-for-llm.md`
- `current-next-step.md`

If a proof file becomes stale or redundant, replace it with a shorter current proof instead of accumulating duplicate logs.

## What this does NOT yet mean

This does **not** yet imply that the whole broader hosted toolchain/runtime is ready for FFmpeg/sqlite/libpng/SPEC.

For the current recommended next milestone, see:
- `current-next-step.md`


