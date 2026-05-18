# Prompt for continuing this Capstone work in a new chat

Use the following prompt as the opening message in a fresh chat.

---

I am continuing work on the Capstone architecture support in the repository:
- `$CAPSTONE_REPO_ROOT`

## Working style / constraints
These are local workspace overlays on top of normal LLVM/Buildroot/Linux/QEMU practices.

1. If you run terminal commands, prefer redirecting output into files under `$CAPSTONE_TMP_ROOT/` and then inspect those files.
2. Be iterative and conservative.
3. Prefer the smallest meaningful next step toward the real goal.
4. Preserve existing style and avoid unrelated refactors.
5. Re-test every completed step at the affected layer.
6. Document non-trivial code concisely, especially protocol layouts, state transitions, ownership rules, branches, and call-sensitive logic.
7. Keep `capstone/agent-handoff/` current when the validated baseline or workflow changes.
8. Never delete `$CAPSTONE_REPO_ROOT/.idea/`.
9. Do not hide nested component repositories from the workspace.
10. History notes must be in English, use `DD-MM-YYYY_HH-MM-SS` filenames, and avoid proper names in titles/filenames.
11. Top-level helper scripts that are not specific to a child repository should live under `capstone/utils/`.
12. After a coherent validated change set, if a commit is appropriate, report exact `git add` / `git commit` commands and prefer a multi-line commit message with a short subject plus a detailed body.
13. Keep manager-facing summaries as local artifacts under `$CAPSTONE_TMP_ROOT/`, not as committed files.

## Read these files first

Read only this minimal startup set before proposing changes:

- `$CAPSTONE_HANDOFF_DIR/README.md`
- `$CAPSTONE_HANDOFF_DIR/current/current-state.md`
- `$CAPSTONE_HANDOFF_DIR/current/current-next-step.md`

Then load deeper files only if the task needs them:

- `$CAPSTONE_HANDOFF_DIR/current/testing-matrix.md`
- `$CAPSTONE_HANDOFF_DIR/current/capstone-agent-test-instructions.md`
- `$CAPSTONE_HANDOFF_DIR/current/stable-file-service-subset.md`
- `$CAPSTONE_HANDOFF_DIR/current/split-host-enclave-strategy.md`
- `$CAPSTONE_HANDOFF_DIR/current/hosted-libc-os-analysis.md`
- `$CAPSTONE_HANDOFF_DIR/current/capstone-backend-status-for-llm.md`
- `$CAPSTONE_HANDOFF_DIR/history/README.md`

## Current verified state

The following is already verified:

1. The LLVM Capstone backend builds the `my_first_domain` sample.
2. Native `ld.lld` support for `EM_CAPSTONE` exists in the current tree.
3. The Buildroot userspace loader accepts the sample domain in the validated path.
4. `capstone/caplifive-buildroot/build/local.mk` is present and keeps the Buildroot image on the local Capstone-enabled Linux/OpenSBI override path.
5. The restored runtime baseline now includes:
   - `capstone/tests/runtime-qemu/run-shared-region-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-file-open-close-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-file-handle-write-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-file-handle-read-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-file-handle-sync-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-file-handle-stat-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-combined-file-object-probe.sh`
   - baseline `null_blk`
   - split `null_blk`
6. The HostCall proofs now cover both payload directions on the same metadata ABI, a reusable handle-based file-object core, an explicit sync boundary after writes, and a narrow stat path for file size/type facts.

## Very important distinction

The validated path today is still the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux user-space.

The preferred near-term direction remains:

- split host-enclave execution,
- shared regions + synchronous multi-round HostCall,
- then a small reusable service surface,
- with `FILE_SYNC` and `FILE_STAT_BASIC` already validated and a narrow handle-based `FILE_TRUNCATE` now chosen as the next semantic,
- only later broader hosted user-space ambitions.

## What to avoid spending time on right now

Unless it directly blocks the active milestone, postpone:

- GISel support,
- cosmetic cleanups,
- pretty disassembly work,
- broad speculative refactors,
- per-libc-symbol HostCall design.

## Expected workflow in the new chat

1. Read the minimal handoff set.
2. Summarize the current verified state briefly.
3. Identify the next smallest meaningful milestone from the current state.
4. Load only the deeper docs needed for that milestone.
5. Implement the minimal justified patch.
6. Rebuild and test.
7. Update the handoff files if the validated state or workflow changed.

When responding, prefer concrete proven facts over assumptions.


