# Current recommended next step

## Immediate milestone - Add the next single BEEBS benchmark

**Goal**: extend the validated BEEBS pattern from `fac`, `insertsort`, `fibcall`,
`cnt`, `bubblesort`, `prime`, `recursion`, `janne_complex`, and `tarai` to
exactly one more deterministic benchmark.

**Why this next**: `tarai` added deeper recursive-call coverage with scalar
global state. BEEBS should keep expanding one small verified benchmark at a time
without introducing a suite runner or performance reporting.

**Recommended candidate**: start with `cover`.

Rationale:
- it is integer-only and deterministic;
- it has deterministic `initialise_benchmark()`, `benchmark()`, and
  `verify_benchmark()` functions;
- it has no benchmark global data that needs the scalar/array accessor workaround;
- it adds dense `switch`/control-flow coverage via `swi10`, `swi50`, and `swi120`;
- it avoids the known floating-point/library-call hazards seen in `sqrt` and the
  benchmarks whose verifier returns `-1`.

**Smallest useful first step**:
- inspect `src/cover/libcover.c` and the generated Capstone assembly,
- copy the existing BEEBS build/host/run pattern conservatively,
- add benchmark-local source wrapping only if the benchmark exposes the same gp-derived
  linear capability reuse issue seen in other global-state BEEBS wrappers,
- keep `fac`, `insertsort`, `fibcall`, `cnt`, `bubblesort`, `prime`,
  `recursion`, `janne_complex`, and `tarai` working as regression gates for the
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
- `bash capstone/benchmarks/beebs/run-beebs-bubblesort.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-prime.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-recursion.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-janne-complex.sh`,
- `bash capstone/benchmarks/beebs/run-beebs-tarai.sh`,
- the focused build/run wrapper for the new single benchmark once introduced.

## Thinking-level rule

Stay at medium thinking while the work remains mechanical or locally debuggable.
If the next benchmark exposes a hard backend/compiler bug, unclear architecture
semantics, or repeated failed runtime debugging where higher thinking looks necessary,
suspend work and tell the user before continuing.

## Candidate caution

Do not pick `sqrt` as the next medium-thinking benchmark: direct Capstone compile currently hits an unsupported softened floating-point library-call path. Benchmarks whose `verify_benchmark()` returns `-1` also remain out of scope for correctness-marker bring-up.

## Remaining backend workarounds

The prologue bug is closed. Keep the remaining backend workarounds in place unless a focused benchmark step proves a root fix. Details: `plans/backend-compiler-fixes.md`.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently switches the image to stock OpenSBI and breaks all runtime proofs.
