# Current recommended next step

## Current BEEBS milestone - 58 benchmarks validated

58 BEEBS benchmarks now pass end-to-end. The most recent addition is `slre`,
validated with `run-beebs-slre.sh`.

## Recent root fixes

Narrow truncating stores from i128 carrier to capability-addressed memory
are fixed:

- `selectLDC_STC` in `CapstoneISelDAGToDAG.cpp` now handles `MemVT = i32/i16/i8`
  truncating stores by emitting `SW`/`SH`/`SB` respectively.  The large-offset
  CIncOffset decomposition is also extended to cover SW/SH/SB.
- This arises when a pointer-difference result (i64 in an i128 any_extend carrier)
  is stored into a narrower integer field (`int len = ptr1 - ptr2`).
- `slre` is the proof benchmark.

The large-offset capability load/store backend blocker is fixed:

- `selectLDC_STC` in `CapstoneISelDAGToDAG.cpp` now handles constant offsets
  > 2047 for `ldc`/`stc` by emitting `CIncOffset(base, offset)` then
  `ldc/stc rd, 0(adjusted)`.
- `sglib-rbtree` was the proof benchmark: its iterator struct pushes
  `equalto` and `subcomparator` past the 2047-byte immediate range.

Pointer arithmetic fixes now cover:

- `ptr - integer` and `ptr + (-offset)` as `cincoffset base, -offset`.
- True `ptr - ptr` by extracting both capability cursors with `lcc ..., 2`,
  subtracting the XLEN cursor values, and sign-extending back to the `i128`
  carrier when needed.
- C pointer subtraction in Clang CodeGen truncates the result to C `ptrdiff_t`
  when the target pointer integer type is wider. This avoids ICmp type
  mismatches when comparing pointer differences on Capstone.
- InstCombine `or disjoint` pointer-carrier additions are lowered as capability
  offset arithmetic when one operand is a known capability base and the other is
  a known integer offset.

Constant-pool lowering for large scalar constants is fixed:

- `lowerConstant` now emits `LOAD:i64(LGA:i128(TargetConstantPool))` directly
  when a large i64 constant is placed in the constant pool, so the final load
  has a capability base rather than a raw integer constant-pool address.
- `lowerConstantPool` returns the capability address (`LGA:i128`) like
  `lowerGlobalAddress`.

`qrduino` and `miniz` both needed benchmark-local source adaptations for static
string pointer data: keep string literals as arrays so benchmark code derives
capabilities from the global symbol rather than loading untagged raw pointers
from data.

`miniz` additionally uses generated scratch sources under
`$CAPSTONE_TMP_ROOT/beebs-build`, strips hosted includes, provides inline libc
stubs, expands/aligned its bump heap, and rounds allocations to 16 bytes.

Verified gates for this milestone:

```bash
source capstone/tests/capstone-test-env.sh
"$CAPSTONE_LLVM_LIT" -sv clang/test/CodeGen/cap-ptr-compare.c
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-beebs-ctl-string.sh
bash capstone/benchmarks/beebs/run-beebs-qrduino.sh
bash capstone/benchmarks/beebs/run-beebs-sglib-rbtree.sh
bash capstone/benchmarks/beebs/run-beebs-miniz.sh
bash capstone/benchmarks/beebs/run-beebs-slre.sh
```

## Remaining viable targets

No clean-add BEEBS target is known. Pick remaining targets only if you are ready
to fix a root issue or carry an invasive source adaptation.

Good next investigations:

- `wikisort`: keep lead-owned.  A scratch pointer-based adaptation under
  `$CAPSTONE_TMP_ROOT/beebs-build` builds, avoids the original stack-cache OOB
  by using a static 512-entry cache, and sorts correctly under native GCC, but
  still fails the Capstone/QEMU correctness marker.  After guarding copy byte
  counts against negative `Range_length(...)` values, the QEMU trap changes
  into a stable wrong-result failure: random test case (`test_case=1`) first
  becomes unsorted at index 6 (`13894 > 12446`).  Direct comparison calls,
  direct field comparison, cursor-integer `memmove` direction checks, and
  widening `Test` fields to `long` did not fix it.  `-O1` is not a workaround:
  it currently hits an unrelated SelectionDAG alias-analysis assertion in
  `APInt::getSExtValue()` during `benchmark`.  Do not add `wikisort` scripts
  until a real QEMU correctness pass exists.
- `trio`: blocked on `va_list` capability storage/copying.
- FP-blocked benchmarks: require a deliberate soft-float/libcall strategy for
  Capstone, not one-off wrappers.

## Blocked (do not retry without root fix)

### Backend crash - other (pre-existing)

- `compress`, `dtoa`, `cubic`: known backend crashes.
- `wikisort`: pointer-based source rewrite is not enough yet; current scratch
  variant fails QEMU correctness after the original OOB is avoided.

### FP-blocked (soft-float libcalls on Capstone)

- `matmult-int` (misleadingly named; uses float matrix)
- `minver`, `ludcmp` - explicit float arithmetic
- `qsort`, `select` - float array comparisons
- `sqrt`, `qurt`, `fasta`, `frac`, `st`, `stb_perlin`, `whetstone` - float
- `newlib-exp`, `newlib-log`, `newlib-mod`, `newlib-sqrt` - math library
- `nbody`, `trio`, `trio-snprintf`, `trio-sscanf` - float / complex format lib

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
bash capstone/benchmarks/beebs/run-beebs-sglib-rbtree.sh
bash capstone/benchmarks/beebs/run-beebs-miniz.sh
bash capstone/benchmarks/beebs/run-beebs-slre.sh
```

## Known backend limitations (document when encountered)

- **memcpy/memmove/memset libcall**: the Capstone backend crashes with null
  symbol name when generating calls to these. Always provide inline stubs
  instead.
- **cincoffset commutative bug**: fixed in lowerADD (isIntegerOffset now covers
  scaled-index GEP; isCapabilityValue distinguishes genuine ldc loads from
  sextloads). edn was the last benchmark blocked by this.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence
silently switches the image to stock OpenSBI and breaks all runtime proofs.
