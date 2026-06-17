# Current recommended next step

## Current BEEBS milestone — 53 benchmarks validated

53 BEEBS benchmarks now pass end-to-end. The most recent confirmed additions are:
- sglib-dllist, sglib-hashtable (38th and 39th — were committed but missed in prior count)
- crc, statemate, nettle-arcfour, nettle-des, aha-mont64, dijkstra
- ctl-stack, ctl-vector (51st and 52nd)
- edn (53rd — unblocked by the cincoffset operand-swap fix in lowerADD)

## Remaining viable targets

None. All remaining unprobed benchmarks are blocked — see Blocked section below.

## Blocked (do not retry without root fix)

### Pointer subtraction (i128 sub — no isel pattern)
- **ctl-string**: `temp - s->string` pointer differences pervasively.
- **qrduino**: Also hits cincoffset commutative bug at -O0; backend crash at -O1.
- **miniz**: `tinfl_decompress` and `tdefl_compress_block` both use pointer
  subtraction pervasively (12+ sites in tinfl/tdefl) to compute buffer offsets
  and byte counts. Not practical to rewrite. Confirmed blocked: `ICmpInst`
  assertion failure comparing `i128` (pointer diff) against integer literals.

### Backend crash — large i128 load constant offset (sglib-rbtree)
- `sglib__rbtree_it_compute_current_elem`: constant offset 2224 exceeds `lc`
  immediate range (12-bit, max 2047). Cannot select `i128 load` node.

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
```

## Known backend limitations (document when encountered)

- **Pointer subtraction (i128 sub)**: subtracting two capability-typed pointers
  generates `i128 sub` with no isel pattern. Avoid in stubs and benchmark adaptations.
- **Large lc offset (>2047)**: loading a capability from a base capability with
  constant offset > 12-bit signed range crashes the backend (sglib-rbtree case).
- **memcpy/memmove/memset libcall**: the Capstone backend crashes with null symbol
  name when generating calls to these. Always provide inline stubs instead.
- **cincoffset commutative bug**: fixed in lowerADD (isIntegerOffset now covers
  scaled-index GEP; isCapabilityValue distinguishes genuine ldc loads from
  sextloads). edn was the last benchmark blocked by this.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` — its absence silently
switches the image to stock OpenSBI and breaks all runtime proofs.
