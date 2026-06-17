# Current recommended next step

## Current BEEBS milestone — 50 benchmarks validated

50 BEEBS benchmarks now pass end-to-end. The most recent additions are:
- crc, statemate, nettle-arcfour, nettle-des, aha-mont64, dijkstra (this session)
- ctl-stack, ctl-vector (this session)

## Remaining viable targets

### miniz (feasible, complex)
`src/miniz/miniz.c` + `src/miniz/miniz_b.c`. No float arithmetic. Needs:
- memcpy/memset/memmove stubs (like ctl-vector)
- assert stub: `#define assert(x) ((void)(x))`
- Strip: `<stdlib.h>`, `<string.h>`, `<assert.h>`, `<stddef.h>`
- The `#include <time.h>` is conditionally compiled; strip `<time.h>` too
- BEEBS provides its own malloc_beebs in miniz.c
- Risk: unknown if backend crashes on the large codebase; try it

### edn (deferred — cincoffset commutative bug)
Build succeeds but runtime fails: `helper_cscincoffset: Assertion 'rs1_v->tag' failed`
in `jpegdct`, `fir_no_red_ld`, `iir1` functions. Fixing requires rewriting those
functions to avoid multiple variable-index array accesses per loop iteration.
Defer unless the root backend bug is fixed.

## Blocked (do not retry without root fix)

### Pointer subtraction (i128 sub — no isel pattern)
- **ctl-string**: `temp - s->string` pointer differences pervasively.
- **qrduino**: Also hits cincoffset commutative bug at -O0; backend crash at -O1.

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
- **cincoffset commutative bug**: backend treats cincoffset as commutative; when
  capability ends up in a higher register than the integer offset, operands are
  swapped → tag fault. Affects any function with multiple variable-index array
  accesses in the same loop iteration.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` — its absence silently
switches the image to stock OpenSBI and breaks all runtime proofs.
