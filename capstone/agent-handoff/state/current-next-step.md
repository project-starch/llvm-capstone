# Current recommended next step

## Current BEEBS milestone — 56 benchmarks validated

56 BEEBS benchmarks now pass end-to-end. The most recent addition is
`sglib-rbtree`, validated with `run-beebs-sglib-rbtree.sh`.

## Recent backend root fixes

The large-offset capability load/store backend blocker is fixed:

- `selectLDC_STC` in `CapstoneISelDAGToDAG.cpp` now handles constant offsets
  > 2047 for `ldc`/`stc` by emitting `CIncOffset(base, offset)` then
  `ldc/stc rd, 0(adjusted)`, matching the existing large-offset pattern for
  integer loads. Regression coverage in `load-store.ll`.
- `sglib-rbtree` was the proof benchmark: its iterator struct has
  `path[128]` of 16-byte capabilities (2048 bytes), pushing `equalto` and
  `subcomparator` past the 2047-byte limit.

The pointer-decrement backend blocker is fixed and validated:

- `ptr - integer` and `ptr + (-offset)` patterns that reach SelectionDAG as
  `sub i128` now lower to `cincoffset base, -offset`.
- Regression coverage lives in `llvm/test/CodeGen/Capstone/ptr-arith.ll`.

The true pointer-difference backend blocker is fixed and validated:

- `ptr - ptr` patterns now lower by extracting both capability cursors with
  `lcc ..., 2`, subtracting the XLEN cursor values, and sign-extending the
  integer result back through the `i128` carrier when needed.
- `ctl-string` is the proof benchmark. Its wrapper keeps the upstream source in
  `/tmp/capstone`, strips hosted includes, provides freestanding libc stubs,
  forces integer `CTL_GROWFACTOR`, and aligns the benchmark bump allocator for
  capability-bearing structs.
- `qrduino` was source-adaptation-local, not a backend root fix. The original
  scratch runtime crashed in `helper_cscincoffsetimm` because a static pointer
  initialized to a string literal was loaded as an untagged scalar and then used
  as the source pointer for `memcpy`. The wrapper keeps the literal as a static
  array, strips hosted includes, provides inline libc stubs, aligns the heap, and
  uses a byte-array verifier to avoid integer bulk-copy corruption.

Verified gates for this milestone:

```bash
source capstone/tests/capstone-test-env.sh
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone/ptr-arith.ll
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-beebs-stringsearch1.sh
bash capstone/benchmarks/beebs/run-beebs-ndes.sh
bash capstone/benchmarks/beebs/run-beebs-ctl-string.sh
bash capstone/benchmarks/beebs/run-beebs-qrduino.sh
bash capstone/benchmarks/beebs/run-beebs-sglib-rbtree.sh
```

Note: one `run-beebs-ndes.sh` attempt timed out after loading `capstone.ko` with
no guest-command output; an immediate rerun passed. Treat that first result as a
transient QEMU smoke timeout unless it reproduces.

## Remaining viable targets

No clean-add BEEBS target is known.

## Blocked (do not retry without root fix)

### Backend crash / invasive pointer-difference users
- **miniz**: `tinfl_decompress` and `tdefl_compress_block` both use pointer
  subtraction pervasively (12+ sites in tinfl/tdefl) to compute buffer offsets
  and byte counts. Re-probe after the pointer-difference fix before deciding
  whether this is still blocked; do not rewrite it heavily.

### Backend crash — other (pre-existing)
- `compress`, `dtoa`, `cubic`: known backend crashes.
- `slre`: Clang frontend PHINode type mismatch (Bug #11).
- `wikisort`: Range struct passed by value throughout (Bug #10, invasive rewrite).

### FP-blocked (soft-float libcalls on Capstone)
- `matmult-int` (misleadingly named; uses float matrix)
- `minver`, `ludcmp` — explicit float arithmetic
- `qsort`, `select` — float array comparisons
- `sqrt`, `qurt`, `fasta`, `frac`, `st`, `stb_perlin`, `whetstone` — float
- `newlib-exp`, `newlib-log`, `newlib-mod`, `newlib-sqrt` — math library
- `nbody`, `trio`, `trio-snprintf`, `trio-sscanf` — float / complex format lib

## Regression gate (run before each new commit)

```bash
source capstone/tests/capstone-test-env.sh
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-beebs-fac.sh
bash capstone/benchmarks/beebs/run-beebs-strstr.sh
bash capstone/benchmarks/beebs/run-beebs-ndes.sh
bash capstone/benchmarks/beebs/run-beebs-expint.sh
bash capstone/benchmarks/beebs/run-beebs-aha-compress.sh
bash capstone/benchmarks/beebs/run-beebs-nettle-cast128.sh
bash capstone/benchmarks/beebs/run-beebs-crc32.sh
bash capstone/benchmarks/beebs/run-beebs-matmult.sh
bash capstone/benchmarks/beebs/run-beebs-ctl-vector.sh
bash capstone/benchmarks/beebs/run-beebs-ctl-string.sh
bash capstone/benchmarks/beebs/run-beebs-qrduino.sh
```

## Known backend limitations (document when encountered)

- **memcpy/memmove/memset libcall**: the Capstone backend crashes with null symbol
  name when generating calls to these. Always provide inline stubs instead.
- **cincoffset commutative bug**: fixed in lowerADD (isIntegerOffset now covers
  scaled-index GEP; isCapabilityValue distinguishes genuine ldc loads from
  sextloads). edn was the last benchmark blocked by this.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` — its absence silently
switches the image to stock OpenSBI and breaks all runtime proofs.
