# Prompt for continuing this Capstone work in a new chat

Application benchmark study: the [Sublet/PoisonCap design](plans/sublet-poisoncap-memory-study.md)
uses two matched pairs for the nested boundary; default CheriBSD on/off remains
separate reference data. The [planner](../experiments/study/README.md) supports
PoisonCap plans but blocks execution qualification pending observed inner-policy
accounting. Twenty host checks pass. Upstream mruby lists passes 4/4 original
arms; both PoisonCap SQLite modes complete the artifact's 20 active phases at
size 1. Twelve phases are commented out in that artifact, and the main
result oracle is missing. These are readiness results, not a memory ranking.

Application memory behavior: [twelve paired workload configurations](../experiments/applications/results/20260927-reuse/README.md)
pass 72/72 attempts. Every recorded Capstone memory phase matches older-QEMU
controls without in-process collection. The results quantify prompt address
reuse and post-release retention, include the large-retained-graph counterexample,
and make no timing or total-RSS claim.

Application memory: [Capstone ports versus default CheriBSD](../experiments/applications/comparison.md) now covers
FFmpeg and mruby with common allocation counters. The original matrix recorded
six Capstone node-capacity failures. The [QEMU node-reuse follow-up](../runtime/tests/application/results/20260927-node-reuse/README.md)
passes all 27 Capstone repeats at the same 65,536-node capacity, including all six
previous failures, using unchanged application binaries. Keep the original data
and larger-node controls separate. These are memory observations, not timings.

Application execution: use the [persistent Linux guest and shared SDK](../runtime/applications.md).
The one-hart QEMU stack verifies trusted fault/preemption return, process-owned
reclamation and 1,008 repeated starts after exhaustion in one boot. Perl uses the
shared SDK; use upstream test runners, not new per-port VM scripts. This requires
the matching pinned QEMU/monitor/driver and does not claim FPGA support or full POSIX.
Perl's complete `t/base` run currently passes eight of nine files; the
remaining failure needs target subprocess creation. See the
[curated result](../ports/perl/musl/results/2026-09-26/base-tests-rebased-qemu.txt).

Allocator trace tooling: [formats, CLI, validation scope and adapter tests](../ports/common/host/port_trace/README.md).

Use the following prompt as the opening message in a fresh chat.

For PostgreSQL allocator work, use `ports/postgres/memory-contexts/README.md`:
the CMake targets port all four pool managers; the legacy scripts remain
AllocSet-only. Do not infer protected consumer-reproducer coverage from that.

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
7. Keep `capstone/docs/` current when the validated baseline or workflow changes.
8. Never delete `$CAPSTONE_REPO_ROOT/.idea/`.
9. Do not hide nested component repositories from the workspace.
10. History notes must be in English, use `DD-MM-YYYY_HH-MM-SS` filenames, and avoid proper names in titles/filenames.
11. Top-level helper scripts that are not specific to a child repository should live under `capstone/utils/`.
12. After a coherent validated change set, if a commit is appropriate, report exact `git add` / `git commit` commands and prefer a multi-line commit message with a short subject plus a detailed body.
13. Keep manager-facing summaries as local artifacts under `$CAPSTONE_TMP_ROOT/`, not as committed files.

## Read these files first

Read only this minimal startup set before proposing changes:

- `$CAPSTONE_HANDOFF_DIR/README.md`
- `$CAPSTONE_HANDOFF_DIR/state/current-state.md`
- `$CAPSTONE_HANDOFF_DIR/state/current-next-step.md`

Then load deeper files only if the task needs them:

- `$CAPSTONE_HANDOFF_DIR/ref/testing-matrix.md`
- `$CAPSTONE_HANDOFF_DIR/ref/capstone-agent-test-instructions.md`
- `$CAPSTONE_HANDOFF_DIR/design/stable-file-service-subset.md`
- `$CAPSTONE_HANDOFF_DIR/design/sqlite-minimal-vfs-path.md`
- `$CAPSTONE_HANDOFF_DIR/design/split-host-enclave-strategy.md`
- `$CAPSTONE_HANDOFF_DIR/design/hosted-libc-os-analysis.md`
- `$CAPSTONE_HANDOFF_DIR/plans/backend-compiler-fixes.md`
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
   - `capstone/tests/runtime-qemu/run-hostcall-file-handle-truncate-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-path-access-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-path-delete-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-combined-file-object-probe.sh`
   - `capstone/tests/runtime-qemu/run-hostcall-all.sh`
   - `capstone/tests/runtime-qemu/run-nullblk-baseline.sh`
   - `capstone/tests/runtime-qemu/run-nullblk-split-io.sh`
   - `capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh`
   - `capstone/tests/runtime-qemu/run-nullblk-all.sh`
   - `capstone/tests/runtime-qemu/run-coremark.sh`
   - `capstone/benchmarks/beebs/run-all-beebs.sh`
   - `capstone/benchmarks/beebs/run-beebs-fac.sh`
   - `capstone/benchmarks/beebs/run-beebs-insertsort.sh`
   - `capstone/benchmarks/beebs/run-beebs-fibcall.sh`
   - `capstone/benchmarks/beebs/run-beebs-cnt.sh`
   - `capstone/benchmarks/beebs/run-beebs-bubblesort.sh`
   - `capstone/benchmarks/beebs/run-beebs-prime.sh`
   - `capstone/benchmarks/beebs/run-beebs-recursion.sh`
   - `capstone/benchmarks/beebs/run-beebs-janne-complex.sh`
   - `capstone/benchmarks/beebs/run-beebs-tarai.sh`
   - `capstone/benchmarks/beebs/run-beebs-cover.sh`
   - `capstone/benchmarks/beebs/run-beebs-duff.sh`
   - `capstone/benchmarks/beebs/run-beebs-levenshtein.sh`
   - `capstone/benchmarks/beebs/run-beebs-jfdctint.sh`
   - `capstone/benchmarks/beebs/run-beebs-fdct.sh`
   - `capstone/benchmarks/beebs/run-beebs-strstr.sh`
6. The HostCall proofs now cover both payload directions on the same metadata ABI, a reusable handle-based file-object core, an explicit sync boundary after writes, a narrow stat path for file size/type facts, a narrow handle-based truncate path for file-size mutation, and the first SQLite-facing path existence/access and path delete proofs.
7. CoreMark PureCap bring-up is complete. All three algorithms (list, matrix, state machine) run and produce validated CRCs ("Correct operation validated."). CoreMark now uses the compiled C domain_main wrapper; the previous per-domain coremark_domain_entry.S prologue workaround is no longer linked. Remaining backend bug workarounds are documented in `$CAPSTONE_HANDOFF_DIR/plans/backend-compiler-fixes.md`.
8. 78 BEEBS benchmarks build and run end to end on the split host/domain
   runtime path, validating correctness markers. The newest are `matmult-float`
   and `whetstone` (added `atan` to the shared soft-float libm), completing the
   soft-float/libm-only FP class; before that `stb_perlin` and the
   `newlib-{sqrt,exp,log,mod}` routines. The canonical full list lives in
   `state/current-state.md`.
9. The 2026-06-09/10 split `null_blk` unload blocker is fixed. The verified
   baseline includes split unload through `run-nullblk-split-rmmod.sh`; use
   `run-nullblk-all.sh`, `run-hostcall-all.sh`, and `run-all-beebs.sh` as the
   aggregate gates, with individual wrappers kept for focused diagnosis. BEEBS
   is serial by default and supports opt-in isolated parallel runs with
   `RUN_ALL_BEEBS_JOBS=N`.

## Very important distinction

The validated path today is still the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux user-space.

Applications run on the delegated runtime (application ABI v2, `runtime/applications.md`),
the only application runtime: each Linux call crosses once, into the launcher's task. The bare
HostCall transport (shared regions + a synchronous multi-round protocol, with the `FILE_*` and
`PATH_*` proofs above) remains only for the S-mode wire probes and the FPGA gates; the musl
runtime's HostCall v0 mode was removed on 2026-09-30.

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
