# dlmalloc thin-shim port: root-cause of the free-path corruption (#78)

**Date:** 2026-07-04
**Outcome:** The chosen approach — *narrow-on-return / re-widen-on-entry shim over
**unmodified** dlmalloc 2.8.6* — is proven **not viable for anything that frees**.
Stock dlmalloc's in-band metadata layout is incompatible with PureCap capability
pointers. Recorded here so we don't re-try the thin-shim path.

## What was built (uncommitted WIP)

- `capstone/benchmarks/rv8/adapted/dlmalloc.c` — vendored dlmalloc 2.8.6
  (public domain), 2-line edit only (`#ifndef` guards on `MFAIL`/`CMFAIL`).
- `capstone/benchmarks/rv8/adapted/cap_heap.c` — the boundary shim:
  `malloc`→`dlmalloc`+`cap_narrow` (SHRINK to `[p,p+n)`); `free`/`realloc`→
  `cap_rewiden` (cincoffset off the retained wide arena cap) then `dlfree`/
  `dlrealloc`; static 16-aligned arena + `cap_morecore` sbrk; derived
  out-of-arena `MFAIL` sentinel (avoids the `(void*)-1` capability forge).
- Compiles EXIT=0 (1 dead forge in `internal_memalign`, unused by malloc/free).

## Empirical result (full RV8 aggregate, `run-all-rv8.sh`)

| benchmark | allocator API used | result |
|-----------|--------------------|--------|
| qsort     | *none* (static array in tail) | PASS (allocator linked but never called) |
| sha512    | malloc only        | **PASS** |
| norx      | malloc only        | **PASS** |
| dhrystone | malloc + **1 free**   | **TIMEOUT (hang)** |
| aes       | malloc + **4 free**   | **TIMEOUT (hang)** |
| miniz     | malloc + 8 realloc + **14 free** | **TIMEOUT (hang)** |
| primes    | malloc + calloc/realloc pulled in | link error (no string lib → memset/memcpy undefined) — separate, trivial |

Perfect split: **every malloc-only benchmark passes; every benchmark that frees
hangs.** The dlmalloc port genuinely works at runtime for the alloc-only path
(sha512/norx prove it) — the break is exclusively the free/bin machinery.

## Root cause (compile-time confirmed, no QEMU needed)

The hang is not a clean capability fault — it is dlmalloc calling `ABORT`
(`CORRUPTION_ERROR_ACTION`/`USAGE_ERROR_ACTION`, both = `ABORT` since
`PROCEED_ON_ERROR=0`), and our `ABORT`=`cap_abort()` is an infinite spin →
TIMEOUT. So dlmalloc's own integrity check fires on the first binned free.

Why: dlmalloc's chunk/bin layout assumes **`sizeof(void*) == SIZE_T_SIZE`**. On
`capstone64` that is false. `_Static_assert` probe (capstone clang):

```c
struct mc { size_t prev_foot; size_t head; struct mc* fd; struct mc* bk; };
sizeof(void*)            == 8   -> FAILS  (a capability pointer is 16 bytes)
sizeof(size_t)           == 8   -> holds
offsetof(struct mc, fd)  == 16  -> holds  (== 2*SIZE_T_SIZE, ok by luck)
offsetof(struct mc, bk)  == 24  -> FAILS  (bk is at 32: a cap is 16 B, 16-aligned)
sizeof(struct mc)        == 32  -> FAILS  (it is 48)
```

dlmalloc's small-bin free list overlays the `smallbins[]` array as fake chunks,
relying on `fd` at `2*SIZE_T_SIZE` and `bk` at `3*SIZE_T_SIZE` (16/24) with the
bin-array stride equal to the pointer size. With 16-byte capabilities that must
stay 16-aligned, `bk` is at **32, not 24**, the chunk struct is **48, not 32**,
and the bin-array stride (16) no longer matches the fd/bk offsets. The moment a
freed chunk is inserted into a bin, the overlay writes/reads the wrong slots →
dlmalloc detects corruption → `ABORT`. Plain `malloc` never touches the bins
(it carves from `top`), which is why alloc-only benchmarks pass.

This is the standard reason stock C allocators need a real CHERI/PureCap port,
not a boundary shim: their in-band metadata stores capability-typed links whose
size/alignment differ from `size_t`, and the hand-computed offset arithmetic
assumes the two are equal. The incompatibility lives **below** our shim, so no
amount of narrow/re-widen at the boundary can fix it.

## Options going forward (decision needed — big direction)

- **A. Real port: address-based metadata.** Store dlmalloc's internal links
  (`fd`/`bk`/`parent`/`child`, `top`/`dv`/`least_addr`/`seg.base`) as `size_t`
  offsets/addresses, re-deriving a real capability from the wide arena cap at
  each dereference (the CheriBSD approach). Correct and keeps a battle-tested
  algorithm, but is a genuine port of dlmalloc internals — not a thin shim.
- **B. Purpose-built free-list allocator with offset metadata.** A compact
  segregated/best-fit free list that stores all metadata as arena-relative
  offsets (no in-band caps by construction → PureCap-safe). "Reinvents" (which
  was pushed back on earlier) but is small, auditable, and correct.
- **C. Keep `rv8_malloc.c` bump allocator** for the benchmark suite (no reuse),
  and treat "real allocator with free" as a separate, later research artifact.

Recommendation: **A** if we want the paper to say "a standard allocator runs
under C1 narrowing," **B** if we want the smallest correct thing we fully
control. Either is a real effort and should be its own scoped step.

## RESOLUTION (2026-07-05): umm_malloc adopted, RV8 7/7

Chose the "existing allocator with non-in-band metadata" path (a refinement of
option B). **umm_malloc** (Ralph Hempel, MIT) keeps all free-list links as
`uint16_t` block *indices*, never capabilities — PureCap-safe by construction.
Vendored under `benchmarks/rv8/adapted/umm/` behind the same `cap_heap.c` boundary
shim (which carried over unchanged); the only port change was padding umm's block
header to 16 B for capability alignment. **RV8 7/7 pass**, including dhrystone/aes/
miniz (the free/realloc-heavy benchmarks that hung here under dlmalloc). See
`design/bounded-heap-allocator-proposal.md` §7a Rev 2. `dlmalloc.c` deleted.

## Repo state left clean

All 7 RV8 build scripts reverted to `rv8_malloc.c` (`git checkout`), so the RV8
gate is green. `cap_heap.c` and `dlmalloc.c` remain untracked for whichever
direction is chosen. `bounded-heap-allocator-proposal.md` §7a (which recorded the
now-falsified "port dlmalloc behind a thin shim" decision) needs a correction
banner pointing here.
