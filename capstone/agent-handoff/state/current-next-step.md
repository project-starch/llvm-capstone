# Current recommended next step

## Immediate milestone - Add a minimal BEEBS runtime wrapper

**Goal**: run the existing BEEBS `fac` `.dom` on the split host/domain runtime path and check only its correctness marker.

**Why this first**: the build-only BEEBS skeleton now exists under `capstone/benchmarks/beebs/` and produces `$CAPSTONE_TMP_ROOT/beebs-build/beebs_fac_capstone.dom`. The next missing piece is the smallest host/runtime wrapper that proves the BEEBS pattern runs end to end before expanding to more benchmarks.

**Smallest useful first step**:
- add a focused `run-beebs.sh` or `run-beebs-fac.sh` entry point for `fac` only,
- reuse the CoreMark split host/domain launch pattern,
- have the domain wrapper report the existing `BEEBS_RET_CORRECT` marker,
- keep success based on correctness only; do not report or optimize performance scores,
- do not expand beyond `fac` until this one benchmark has a stable build and run path.

**Test expectations**:
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`,
- `bash capstone/tests/runtime-qemu/run-coremark.sh`,
- `bash capstone/benchmarks/beebs/build-beebs-fac-capstone.sh`,
- the focused BEEBS runtime wrapper once introduced.

## Remaining backend workarounds

The prologue bug is closed. Keep the remaining backend workarounds in place unless a focused benchmark step proves a root fix. Details: `plans/backend-compiler-fixes.md`.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently switches the image to stock OpenSBI and breaks all runtime proofs.
