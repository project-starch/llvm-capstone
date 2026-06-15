# Current recommended next step

## Immediate milestone - Continue batched BEEBS bring-up

**Goal**: extend the validated BEEBS pattern from the current 21 benchmarks by
probing and adding another batch of 5-8 deterministic benchmarks.

**Why this next**: the previous batch validated the low-token workflow. Five
benchmarks were added with shared simple-benchmark helpers and thin per-benchmark
entry points:

- `sglib-arraybinsearch`
- `sglib-queue`
- `sglib-listinsertsort`
- `sglib-listsort`
- `expint`

Use the same probe-first workflow for the next batch: classify candidates with
temporary `/tmp/capstone` build/run artifacts, add only cheap passers, and defer
hard failures instead of debugging them inside the batch.

## Batch rules

- Batch size target: 5-8 benchmarks.
- Keep committed run entry points per benchmark; shared helpers are acceptable for
  simple one-source benchmarks.
- Keep fetched BEEBS sources under `$CAPSTONE_TMP_ROOT`; do not vendor sources or
  add submodules.
- Success is correctness marker only; do not report or optimize performance.
- Do not introduce a broad permanent suite runner yet.
- If a candidate exposes a backend/compiler/runtime bug or would need higher
  thinking, skip it and record the failure class.

## Recommended candidate pool

Probe these next, stopping once 5-8 cheap passers are available:

- `aha-compress`
- `nettle-md5`
- `nettle-cast128`
- `slre`
- `matmult`
- `mergesort`
- `nbody`
- `trio`

Prefer candidates with real `verify_benchmark()` implementations. Continue to
avoid benchmarks whose verifier returns `-1` and floating-point-heavy benchmarks
unless they pass the cheap probe without backend work.

## Deferred from the previous probe batch

- `stringsearch1`: backend instruction selection failure in `prep1`.
- `crc32`: builds and runs but returns the wrong correctness marker.
- `sglib-rbtree`, `aha-mont64`, `dijkstra`, `edn`, `ctl-string`, `qrduino`,
  `nettle-arcfour`, `ludcmp`, `nettle-des`, `statemate`: compile-time backend
  crashes or non-trivial source adaptation required.

## Test expectations

For the next committed batch:

- each newly added `run-beebs-*.sh`
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`
- `bash capstone/tests/runtime-qemu/run-coremark.sh`
- focused existing BEEBS regressions: `fac`, `strstr`, `ndes`, and one benchmark
  from the latest batch, preferably `run-beebs-expint.sh`

## Thinking-level rule

Stay at medium thinking while the work remains mechanical or locally debuggable.
If a benchmark exposes a hard backend/compiler bug, unclear architecture
semantics, or repeated failed runtime debugging where higher thinking looks
necessary, suspend work and tell the user before continuing.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently
switches the image to stock OpenSBI and breaks all runtime proofs.
