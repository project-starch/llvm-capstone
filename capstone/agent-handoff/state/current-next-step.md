# Current recommended next step

## Immediate milestone - Add the next BEEBS benchmark batch

**Goal**: extend the validated BEEBS set with new benchmarks from the candidate
pool below.

**Why this next**: the source-adaptation cleanup is complete. All Capstone-specific
BEEBS C is now in real files under `capstone/benchmarks/beebs/adapted/` rather
than embedded in shell heredocs. The next step is resuming new benchmark bring-up.

## Candidate pool

- `aha-compress`
- `nettle-md5`
- `nettle-cast128`
- `slre`
- `matmult`
- `mergesort`
- `nbody`
- `trio`

## Deferred from the previous probe batch

- `stringsearch1`: backend instruction selection failure in `prep1`.
- `crc32`: builds and runs but returns the wrong correctness marker.
- `sglib-rbtree`, `aha-mont64`, `dijkstra`, `edn`, `ctl-string`, `qrduino`,
  `nettle-arcfour`, `ludcmp`, `nettle-des`, `statemate`: compile-time backend
  crashes or non-trivial source adaptation required.

## Test expectations

For each new benchmark commit:

- the new `run-beebs-<name>.sh`
- `"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone`
- `bash capstone/tests/runtime-qemu/run-coremark.sh`
- focused existing BEEBS regressions: `fac`, `strstr`, `ndes`, `expint`

## Thinking-level rule

Stay at medium thinking while the work remains mechanical or locally debuggable.
If a benchmark exposes a hard backend/compiler bug, unclear architecture
semantics, or repeated failed runtime debugging where higher thinking looks
necessary, suspend work and tell the user before continuing.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence silently
switches the image to stock OpenSBI and breaks all runtime proofs.
