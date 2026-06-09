# Current recommended next step

## Immediate milestone - Add BEEBS `bubblesort`

**Goal**: extend the validated BEEBS pattern from `fac`, `insertsort`, `fibcall`,
and `cnt` to exactly one more deterministic benchmark: `bubblesort`.

**Why this first**: `cnt` now covers deterministic verification with global matrix
and scalar state. `bubblesort` is another small verified integer benchmark with a
global array, so it should extend the same pattern without introducing a suite
runner or performance reporting.

**Smallest useful first step**:
- inspect `bubblesort` source and generated Capstone assembly for global array
  accesses,
- copy the existing BEEBS build/host/run pattern conservatively,
- add benchmark-local source wrapping only if `bubblesort` exposes the same gp-derived
  linear capability reuse issue seen in `insertsort`,
- keep `fac`, `insertsort`, `fibcall`, and `cnt` working as regression gates for the
  BEEBS path,
- keep success based on correctness only; do not report or optimize performance scores,
- do not introduce a broad BEEBS suite runner yet.

**Test expectations**:
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`,
- `bash capstone/tests/runtime-qemu/run-coremark.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-fac.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-insertsort.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-fibcall.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-cnt.sh`,
- the focused build/run wrapper for the new single benchmark once introduced.

## Thinking-level rule

Stay at medium thinking while the work remains mechanical or locally debuggable.
If the next benchmark exposes a hard backend/compiler bug, unclear architecture
semantics, or repeated failed runtime debugging where high thinking looks necessary,
suspend work and tell the user before continuing.

## Remaining backend workarounds

The prologue bug is closed. Keep the remaining backend workarounds in place unless a focused benchmark step proves a root fix. Details: `plans/backend-compiler-fixes.md`.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently switches the image to stock OpenSBI and breaks all runtime proofs.
