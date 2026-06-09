# Current recommended next step

## Immediate milestone - Add BEEBS `prime`

**Goal**: extend the validated BEEBS pattern from `fac`, `insertsort`, `fibcall`,
`cnt`, and `bubblesort` to exactly one more deterministic benchmark: `prime`.

**Why this first**: `bubblesort` now covers deterministic verification with a
global array and benchmark-local source wrapping. `prime` is a small scalar-heavy
verified benchmark that adds modulo/division coverage without introducing a suite
runner or performance reporting.

**Smallest useful first step**:
- inspect `prime` source and generated Capstone assembly for scalar global accesses
  and division/modulo lowering,
- copy the existing BEEBS build/host/run pattern conservatively,
- add benchmark-local source wrapping only if `prime` exposes the same gp-derived
  linear capability reuse issue seen in other global-state BEEBS wrappers,
- keep `fac`, `insertsort`, `fibcall`, `cnt`, and `bubblesort` working as
  regression gates for the BEEBS path,
- keep success based on correctness only; do not report or optimize performance scores,
- do not introduce a broad BEEBS suite runner yet.

**Test expectations**:
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`,
- `bash capstone/tests/runtime-qemu/run-coremark.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-fac.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-insertsort.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-fibcall.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-cnt.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-bubblesort.sh`,
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
