# CoreMark 1.01 matrix — silicon-ladder rung 7 (QEMU, validated)

**Date:** 2026-07-24
**Lane:** B (per `plans/coremark-ladder-task-B.md`)
**Result:** CoreMark's matrix benchmark runs as a silicon-ladder rung in a pure-cap
domain on QEMU and returns crc16 `14343`, matching a native `cc -O0` oracle, with
the static gate clean (`cjalr=0`, `ldc-gp=1`). Marker
`__CAPSTONE_LADDER_COREMARK_MATRIX_PASSED__`.

## Files (all under `capstone/tests/runtime-qemu/silicon-ladder/`)
- `coremark_matrix_kernel.h` — amalgamated single-TU kernel: upstream CoreMark
  v1.01 `core_matrix.c` compute + `core_util.c` crc16, verbatim.
- `coremark_matrix_app.c` — `domain_main(res, func)` → `*res = crc`.
- `coremark_matrix_host.c` — native oracle printing the same crc.
- `run-coremark-matrix-qemu.sh` — dedicated wrapper pinning `DOMAIN_OPT_LEVEL=-Os`
  (see below), delegates to the generic `run-ladder-qemu.sh coremark_matrix`.

## Two design decisions and why

### 1. Matrix algorithm only (not the full three-algorithm CoreMark)
CoreMark's list and state CRCs are **pointer-size-dependent**: the benchmark packs
pointers into `list_head` nodes, and at `sizeof(void*)=16` (capability) vs 8 the
node count and traversal differ, so the full-CoreMark CRCs differ from a native
build (see `design/coremark-purecap.md`; the three algorithms are also entangled —
matrix/state are driven from inside the list mergesort comparator `cmp_complex`,
seeded by the list's running CRC). A native `cc` oracle could therefore never fold
the same value for list/state.

The **matrix** kernel operates purely on `ee_s16`/`ee_s32` integer matrices, so its
crc16 is pointer-size-**independent** — the domain and a native host compute the
identical checksum by construction. This is the same oracle contract the crc32 rung
uses. It is driven standalone with CoreMark's real validation-run matrix parameters
(blksize 666 → N=9, init seed 0, bench val 0x66), not through the list, so the value
(`14343`) is a deterministic standalone-matrix crc16, not the canonical entangled
`crcmatrix`. The compute chain is verbatim: `matrix_add_const` → `matrix_mul_const`
→ `matrix_mul_vect` → `matrix_mul_matrix` → `matrix_mul_matrix_bitextract`, each
folded through `matrix_sum` + `crc16`.

Two PureCap adaptations, both `#ifdef __capstone`-guarded so the native oracle
compiles the identical math:
- `align_mem` omitted: upstream builds a pointer from an integer (drops the cap tag
  on PureCap); here the block is a static 16-aligned array and every A/B/C offset is
  a multiple of 4, so the upstream align is provably the identity — a bare pointer
  is correct and keeps the tag.
- `CAPSTONE_DELIN(A)` after deriving A from the gp-delivered block, so `B = A + N*N`
  (cincoffset rd≠rs1) does not consume the LINEAR cap. No-op on native.

### 2. Built `-Os`, not the ladder default `-O0`
The silicon image pins all globals at `base+0x1000` (`link-gpfree.ld`), so domain
`.text` must fit the 4 KiB PCC window. CoreMark's matrix kernel is **~4.7 KiB** at
`-O0` and overflows (link error: `.text` `[0x10000,0x1141B]` overlaps `.bss` at
`0x11000`). It is **~1.5 KiB at `-Os`** and fits with room to spare. `-Os` is safe
here: the kernel is almost entirely integer math (one gp-delivered global,
delinearised once), and the oracle-match assertion catches any optimisation-induced
miscompile loudly. A benchmark at `-O0` is not a meaningful CoreMark anyway. `-Os`
is pinned in the wrapper so a plain re-run reproduces it. (`-O0` was the ladder
default only to dodge the LINEAR cap-sink bugs that bite *cap-heavy* code.)

## Environment finding (not specific to this rung)
The `-b` `clang` binary is stale (Jul-10) and predates the merged
`-capstone-gp-captable` flag — so the whole silicon ladder is unbuildable with the
in-tree `-b` toolchain until it is rebuilt. Validated here by pointing
`CAPSTONE_LLVM_BUILD_DIR` at the sibling checkout's clang (built 2026-07-24, knows
the flag) driving the `-b` QEMU/monitor/rootfs — confirmed consistent by first
re-running the existing `matmult_int` rung green. Also: `-b`'s LLVM build dir had
been reconfigured Jul-22 to a static/clang-disabled config; restored to the correct
shared + `clang;lld` config (build.ninja regenerated), so a future `cmake --build`
produces a current clang. The rebuild itself (~4866 debug actions) was deferred.

## Reproduce
```bash
source capstone/tests/capstone-test-env.sh
# until the -b clang is rebuilt, borrow a clang that knows -capstone-gp-captable:
export CAPSTONE_LLVM_BUILD_DIR=<checkout-with-current-clang>/llvm/cmake-build-debug
bash capstone/tests/runtime-qemu/silicon-ladder/run-coremark-matrix-qemu.sh
# -> __CAPSTONE_LADDER_COREMARK_MATRIX_PASSED__ (retval = 14343)
```
