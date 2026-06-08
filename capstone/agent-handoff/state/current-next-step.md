# Current recommended next step

## Immediate milestone - Add the second BEEBS benchmark

**Goal**: extend the validated BEEBS pattern from `fac` to exactly one more tiny deterministic benchmark.

**Why this first**: `capstone/benchmarks/beebs/run-beebs-fac.sh` now builds and runs `fac` end to end on the split host/domain runtime path. The next useful step is proving the pattern generalizes without creating a full suite runner.

**Smallest useful first step**:
- add `insertsort` only; it is a small deterministic array benchmark with no external data files,
- extend the existing BEEBS build/host/run pattern conservatively,
- keep `fac` working as the regression gate for the BEEBS path,
- keep success based on correctness only; do not report or optimize performance scores,
- do not introduce a broad BEEBS suite runner yet.

**Test expectations**:
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`,
- `bash capstone/tests/runtime-qemu/run-coremark.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-fac.sh`,
- the focused `insertsort` build/run wrapper once introduced.

## Remaining backend workarounds

The prologue bug is closed. Keep the remaining backend workarounds in place unless a focused benchmark step proves a root fix. Details: `plans/backend-compiler-fixes.md`.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently switches the image to stock OpenSBI and breaks all runtime proofs.
