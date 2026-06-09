# Current recommended next step

## Immediate milestone - Add one more BEEBS benchmark

**Goal**: extend the validated BEEBS pattern from `fac` and `insertsort` to exactly one more tiny deterministic benchmark.

**Why this first**: both `capstone/benchmarks/beebs/run-beebs-fac.sh` and `capstone/benchmarks/beebs/run-beebs-insertsort.sh` now build and run end to end on the split host/domain runtime path. The next useful step is proving the pattern generalizes one benchmark at a time without creating a full suite runner.

**Smallest useful first step**:
- pick one small deterministic benchmark such as `cnt`, `crc`, or `fibcall`, after a quick source inspection,
- copy the existing BEEBS build/host/run pattern conservatively,
- keep `fac` and `insertsort` working as regression gates for the BEEBS path,
- keep success based on correctness only; do not report or optimize performance scores,
- do not introduce a broad BEEBS suite runner yet.

**Test expectations**:
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`,
- `bash capstone/tests/runtime-qemu/run-coremark.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-fac.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-insertsort.sh`,
- the focused build/run wrapper for the new single benchmark once introduced.

## Remaining backend workarounds

The prologue bug is closed. Keep the remaining backend workarounds in place unless a focused benchmark step proves a root fix. Details: `plans/backend-compiler-fixes.md`.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently switches the image to stock OpenSBI and breaks all runtime proofs.
