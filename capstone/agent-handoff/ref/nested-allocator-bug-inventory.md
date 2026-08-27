# Nested-allocator bug inventory

*Bugs in the METADATA HANDLING of allocators that sit on top of malloc, or that are
the malloc. Compiled 2026-08-27 on branch `capstone-nested-allocator-bugs`.*

## How this differs from `paper-bug-inventory.md`

That file is organised by the **sharing** taxonomy: a host lends an object to a
scripting binding and the lifetime ends on the wrong side. Its allocator row has 13
specimens, almost all from Lua bindings, and **zero CVEs**.

This file is a different axis: the allocator's **own** bookkeeping is corrupted --
a block header, a free-list link, a size-class index, a bitmap. That is the class a
capability machine has an interesting chance of catching, because the corrupted word
is one the allocator itself will later dereference.

The two do not overlap. Nothing here is in that file, and a set of real CVEs is
new evidence for a row that currently has none.

---

## READ THIS BEFORE BELIEVING ANY VERDICT

Three ways this measurement can produce a clean result that means nothing. Each is
recorded because it was found while compiling the list, not after a wrong claim.

**1. MEMSYS5 rounds every request up.** `memsys5Roundup()` returns `szAtom`,
`szAtom*2`, then `iFullSz` in steps of four with a `/2` refinement (read in
`sqlite3-capstone.c:29830`). A small overflow therefore lands **inside the same
block**. If the capability handed back by `sqlite3_malloc` carries the BLOCK's
bounds rather than the REQUESTED size, roughly half of the SQLite entries below are
invisible and will be scored "not caught". Narrow the capability to the request, and
negative-test that narrowing before trusting a single verdict.

**2. TLSF's own checker crashes instead of reporting.** `tlsf_check()` segfaults on a
deliberately corrupted header rather than returning non-zero. A 4M-operation random
fuzz over upstream TLSF found nothing, and that result says nothing about TLSF.

**2b. GitHub's contents API caps a directory listing at 1000 entries.** A bulk `ls`
of `oss-fuzz/projects/` therefore returns a false "not found" for projects that are
present. The per-project check is the reliable form, and it is what the negative
result below rests on.

**3. A 404 is not an absence.** `cveawg.mitre.org` returned 404 for CVE-2026-54913,
which is entirely real -- GitHub is the CNA and the record had not propagated, and
GitHub's *global* advisory endpoint does not mirror *repository* advisories either.
Use `capstone/tests/check-cve-ids.py`, which says "not in MITRE's published set", and
enumerate `gh api /repos/<owner>/<repo>/security-advisories` before concluding
anything.

---

## Ready to reproduce

Everything here needs nothing we lack: no OS, no threads, no MMU, no filesystem, no
JIT. `[V]` marks a claim re-checked against the primary source by the main session,
not only reported.

### WAMR -- its own EMS allocator

| # | Ref | Component | Metadata corrupted | Note |
|---|---|---|---|---|
| 1 | PR 2279, fix `4fcc05617865` | `ems_alloc.c` `gc_realloc_vo_internal` | `pinuse` bit never set on the new `hmu_next`, so a later free **coalesces backwards into a live object** | **[V]** a self-contained C reproducer is already in our tree at `samples/mem-allocator/main.c`; the fix `hmu_mark_pinuse(hmu_next)` is present at `ems_alloc.c:829`, so this is reproduced by REVERTING one line, with the fix as the control |
| 2 | issue 2136, fix `5c497e5a1447` | `ems_alloc.c` `remove_tree_node`, `gci_add_fc` | free-list tree nodes not 8-byte aligned on **riscv64** | the umbrella issue is still open; our exact target |
| 3 | PR 428, fix `91b9458ebd42` | `ems_alloc.c` `remove_tree_node` | no `hmu_is_in_heap` check on node or parent: a corrupted free-tree node is a write-anywhere primitive | pairs with #1 |
| 4 | PR 788 / PR 4862 | `ems_kfc.c` `gc_migrate`, `gci_dump` | heap walked by `cur += hmu_get_size(cur)` trusting an attacker-influenced size | **[V]** `BH_ENABLE_GC_CORRUPTION_CHECK` defaults to 1 (`core/config.h:371`); the 2021 fix had to be made again in 2025 because it was compiled out |
| 5 | PR 4161 / 4546 | `ems_kfc.c` `adjust_ptr`, `gc_update_threshold` | signed overflow in pointer fixup; 32-bit multiply in the GC threshold | pool-placement dependent, which a freestanding target controls |

### WAMR and wasm3 -- loader and interpreter

| # | Ref | Component | Class | Note |
|---|---|---|---|---|
| 6 | **CVE-2026-54913** | `wasm_loader.c` `load_init_expr` | uninitialised read, host stack leaked to the guest | **[V]** advisory read: any byte accepted as heap type when `WASM_ENABLE_GC=0`, **our configuration**; `cur_value.ref_index = NULL_REF` writes 4 of 16 union bytes. Our pin is 2.4.3, vulnerable is `<= 2.4.4`, **no patched version exists**, and the line is at `wasm_loader.c:1013`. 90-byte module in the advisory |
| 7 | **CVE-2024-34250** | `wasm_loader.c` `wasm_loader_check_br` | 4-byte heap OOB write via `br_table` | **[V]** MITRE names the same function and file; **[V]** the function has zero enclosing `#if` at line 10954, so it is in our classic-interpreter build. PoC attached |
| 8 | **CVE-2023-48105** | `wasm_loader.c` `wasm_loader_prepare_bytecode`, `load_init_expr` | 1-byte heap OOB read, missing `CHECK_BUF` | **[V]** identifier confirmed. Truncated module ending in a block/SIMD prefix |
| 9 | issue 2586, fix `6382162711a9` | `wasm_loader.c` `wasm_loader_ctx_destroy` | **double free** | inline WAT PoC. UNVERIFIED which interpreter mode the original was in |
| 10 | wasm3 issue 570 | `m3_compile.c` `Compile_BranchTable` | int overflow to short alloc to OOB write | **still live in wasm3 main**; guarded only by an assert compiled out in release. PoC and generator attached |
| 11 | wasm3 **CVE-2022-28966** | `m3_code.c` `NewCodePage` | heap OOB write | **[V]** identifier confirmed |
| 12 | wasm3 issue 485, fix `f710292fb87f` | `m3_compile.c` `DeallocateSlot` | heap OOB read via an unallocated slot index | |

### SQLite -- mem5, lookaside, sorter

| # | Ref | Component | Metadata / class | Trigger |
|---|---|---|---|---|
| 13 | check-in `b3296267fb67b9f5` | `mem5.c` `memsys5MallocUnsafe` | reads `aiFreelist[31]`, one int past the array, because the subscript is evaluated before the bound test | exhaust the arena. **The only bug IN memsys5**, and on a 256 KiB arena its precondition is our steady state |
| 14 | **CVE-2025-29088** | `main.c` `setupLookaside` | `sz*nBig` not cast to 64-bit, slot carving writes a `pNext` every 128 bytes past the buffer | **[V]** MITRE verbatim. `sqlite3_db_config(db, SQLITE_DBCONFIG_LOOKASIDE, pBuf, 1200, 2500000)` with **our own** buffer. The catching assert is compiled out by `-DNDEBUG` |
| 15 | check-in `8b7a7fcf62e5c274` | `vdbesort.c` `sqlite3VdbeSorterInit` | 8-byte over-read of the source KeyInfo | three plain statements on `:memory:`; needs `MEMSTATUS=0`, which we already set. Cleanest repro in the set |
| 16 | check-in `a89b38605661e36d` | `build.c` `sqlite3EndTable` | lookaside nodes grafted into the schema, heap corruption at close | a generated column whose expression fails name resolution. Needs lookaside enabled |
| 17 | **CVE-2021-20227** | `select.c` `havingToWhereExprCb` | read-after-free | **[V]** identifier confirmed. Regression test `having.test` 5.1 |
| 18 | **CVE-2020-13871** | `select.c` `resetAccumulator` | use-after-free | **[V]** identifier confirmed. Regression test `window1.test` 55.1 |
| 19 | check-in `0e4789860b81c31d` | `select.c` `sqlite3Select` | UAF: `pDistinct` deleted while `pWInfo` still reads it | `distinctagg.test` 7.0 |
| 20 | check-in `92893b7980cbb0c6` | `select.c` `convertCompoundSelectToSubquery` | **double free**: `pWinDefn` not cleared, parent and child both own it | `window1.test` 35.0/35.1. The only true double free here, and both frees hit our allocator |
| 21 | check-in `807643c596b2315f` | `build.c` `resizeIndexObject` | under-allocation, `sizeof(LogEst)` omitted | WITHOUT ROWID table, many columns, NATURAL JOIN with itself |

Our tree pins `SQLITE_VERSION=3530300` **[V]**, so all of these are already fixed;
`fetch-sqlite.sh` is parameterised, so an older amalgamation is one variable.

### Lua

| # | Ref | Component | Class | Trigger |
|---|---|---|---|---|
| 22 | **CVE-2020-24370** | `ldebug.c` `lua_getlocal` | negation overflow to OOB | **[V]** identifier confirmed. `print(debug.getlocal(1, 2^31))` -- one line, no GC, no metatables |
| 23 | **CVE-2020-15889** | `lgc.c` `youngcollection`, `markold` | OOB read + **UAF**: a finalized OLD1 object moves to the head of `allgc`, which `markold` never visits | **[V]** identifier confirmed. Short script in the strand report |
| 24 | **CVE-2020-24371** | `lgc.c` barriers during sweep | **UAF**: `gray` and `grayagain` mixed, an old table dropped | **[V]** identifier confirmed |
| 25 | **CVE-2021-44964** | `lgc.c` finalizer reentrancy | UAF | **[V]** identifier confirmed. Fix `0bfc572e51d9` proven by MATCHED PAIR: `commit^` faults in `separatetobefnz`, `commit` clean. Note two independent runs found 5.4.0 clean, contradicting NVD's lower bound |
| 26 | **CVE-2020-15888** | stack resize during GC | heap UAF in `luaD_call` | **[V]** identifier confirmed |
| 27 | **CVE-2022-28805** | `lparser.c` `singlevar` | OOB read via `<const> _ENV` | **[V]** identifier confirmed. Caveat: only the reported blob faults; the readable minimal form errors cleanly, so it demonstrates the shape, not the trigger |

### Standalone and libc allocators

| # | Ref | Component | Metadata corrupted | Note |
|---|---|---|---|---|
| 28 | Contiki-NG, introduced `470c5a875e26`, removed `253882f5751d` | `os/lib/heapmem.c` `IN_HEAP` | **[V]** read verbatim at `release/v5.1`: the macro's outer paren closes after the second condition, so `!IN_HEAP(p)` is `!(A && B) && C`. Under-range pointers are rejected; **every over-range and foreign pointer passes** into `free_chunk` and writes a chunk header at a caller-controlled address | affects v4.8 to v5.1. A one-character bug with a write-anywhere consequence |
| 29 | TLSF issue 22 follow-up, **OPEN** | `tlsf_realloc` `block_split` | `block_split(block, 0)` lands the remainder **on the block's own header**, then writes a forged free header into the first 8 bytes of the live payload and inserts it into `blocks[fl][sl]` | no upstream fix; espressif carries one |
| 30 | TLSF issue 35, **OPEN** | `tlsf_create_with_pool` | `tlsf_create` takes no size, so a ~6.5 KB `control_t` is written past an undersized buffer | no upstream fix |
| 31 | umm_malloc issue 62, fix `da41963ae859` | `umm_blocks()` | `2 + size / UMM_BLOCKSIZE` truncated to `uint16_t`: silent under-allocation, then the neighbours' **block indices** are destroyed | affects <= v1.1.0 |
| 32 | tinyalloc issue 15, **OPEN** | `alloc_block()` | two `Block` descriptors on `used` carry the **same `addr`**; freeing one leaves the other live and dangling | PR unmerged |
| 33 | dlmalloc, **LIVE in 2.8.6** | `dlpvalloc()` | `bytes + pagesz - 1` wraps, the mask yields a tiny size, `MAX_REQUEST` never sees the original | no upstream fix |
| 34 | musl oldmalloc | `memalign` | writes 1 over the previous chunk's **footer** | the interesting question is whether narrowing memalign's own capability catches it |
| 35 | FreeRTOS `heap_2` | double free | produces a self-loop in the free list | three lines |
| 36 | RIOT `pkg/tlsf/contrib/native.c`, **UNFIXED** | `calloc` | `count*bytes` overflow, no check; the 2021 fix patched only the newlib arm | |
| 37 | TLSF issue 22, **OPEN** | `adjust_request_size()` tlsf.c:492 | `align_up(SIZE_MAX, ALIGN_SIZE)` wraps to 0, passes `< block_size_max`, so `adjust` becomes `block_size_min`: the block's **`size` field** says minimum while the caller believes it owns `SIZE_MAX` | **reproduced by the strand**: `tlsf_malloc(t, SIZE_MAX)` returns a 24-byte block on 64-bit. **espressif's fork is still vulnerable too**; ESP-IDF is shielded only by a caller-side bound |
| 38 | Zephyr issue 90306, fix `811302e6d261` | `sys_heap` `size_too_big()` | the check divides before the header is added and rounded, so `bucket_idx()` indexes **past `h->buckets[]`**; `b->next` is then heap payload used as a chunk id, and the unlink writes `FREE_PREV`/`FREE_NEXT` at an arbitrary offset | affects v2.5.0 to v4.1.x. Trigger: `CONFIG_HEAP_MEM_POOL_SIZE=2048`, then `k_malloc(2033)` |
| 39 | **CVE-2020-13603** | Zephyr `sys_mem_pool_alloc` | integer overflow | **[V]** identifier confirmed. `malloc(0xffffffff)` returns a successful **7-byte** allocation |
| 40 | umm_malloc issue 77, fix `08df3497cb0f` | `umm_multi_init_heap` | same 15-bit block index as entry 31, at init time: a heap over 256 KiB corrupts itself | fixed 2026-06-24, **no release tag yet** |
| 41 | RTEMS issue 1258, fix `493e405cac79` | `_Heap_Allocate_aligned()` | the user address is computed downward assuming the upper half of a split becomes used, but the realloc optimisation made the **lower** half used: the returned pointer lands inside a block still on the free list, so user writes hit its `prev_size`, its `size` word with `HEAP_PREV_USED`, and its free-list links | affects 4.7 |

**41 entries needing nothing we lack**, against a target of 25 to 30.

---

## Out of scope, with the reason

Recorded so nobody spends time rediscovering it.

* **Fast interpreter only**, and we build the classic one: CVE-2026-54912, WAMR
  issues 3513, 3580, 3514, 3137, CVE-2025-64713.
* **JIT / AOT / WASI / sockets**: CVE-2025-58749, CVE-2026-54914, CVE-2025-43853,
  CVE-2025-54126.
* **Needs a feature we compile out**: CVE-2025-64704 (SIMD), WAMR issue 4942 (GC),
  issue 4935 (MEMORY64 plus multi-memory), SQLite `PRAGMA foreign_key_check`
  (foreign keys), CVE-2020-13434 (floating point).
* **Too large for a 256 KiB arena**: CVE-2025-6965 (>32767 aggregate terms),
  CVE-2025-3277 (~17 MB separator), musl mallocng M1 (~2 GiB of address space).
* **LuaJIT entirely**, and NOT because of the JIT: its interpreter is a hand-written
  DynASM VM with no RISC-V backend in `lj_arch.h`. Issue 1471 is a genuine heap
  overflow in `lj_state_growstack` that reproduces with `-joff`, so if a RISC-V VM
  port ever lands this class becomes reachable.
* **C++ / STL**, per the existing feasibility rules in `paper-bug-inventory.md`.
* **Wrong word size or needs mmap**: dlmalloc's `binmap[BINMAPSIZE]` misdeclaration
  (v2.7.0/2.7.1) needs 32-bit -- `NBINS 96` gives `BINMAPSIZE 3` and `++idx` reads
  `binmap[3]`; we are riscv64. dlmalloc's `internal_memalign` brace error
  (v2.8.0/2.8.1) needs mmap.

## Negative results, positive-controlled

* **No OSS-Fuzz project exists** for tlsf, umm_malloc, o1heap, tinyalloc, dlmalloc,
  zephyr, contiki-ng, rtems, riot or newlib. Controls: `tcmalloc` present, a bogus
  name absent. There are no OSS-Fuzz IDs to cite for these.
* **No GHSA** for `mattconte/tlsf`, `rhempel/umm_malloc`, `pavel-kirienko/o1heap`,
  `thi-ng/tinyalloc`, `espressif/tlsf`. Controls: contiki-ng has 31, zephyr 216.
* **o1heap has no documented bugs.** Zero CVEs, zero GHSAs, no post-release fix in
  `o1heap.c`; both reports against it were caller-side corruption. That is the
  truthful answer for that project, not a gap in the search.
* **RTEMS and Zephyr do not use TLSF.** RTEMS has zero TLSF commits (control: 59 for
  `*heapallocate*`) and its own first-fit `Heap_Control`; Zephyr's `sys_heap` uses a
  single-level bitmap. RIOT is the only one of the three with a real TLSF.

## Verification status

Every CVE identifier quoted above was checked with
`capstone/tests/check-cve-ids.py`, which carries a positive and a negative control.
**All exist; no fabrications were found** in 88 raw candidates. That is measured,
not assumed -- and it matters, because `paper-bug-inventory.md` already carries a
list of four fabricated CVE numbers, and sqlite.org itself flags six more
(CVE-2026-51296, -51297, -51300, -51302, -51303, -51304) as unreproducible and
probable AI output.

Not verified, and marked as such above: the interpreter mode for WAMR issue 2586;
library-level PoCs for CVE-2020-15358 and CVE-2025-7458; several dlmalloc changelog
entries that were read rather than reproduced; TLSF issues 36 and 9.

## Where to start

1. **Entry 1**, the EMS realloc pinuse bit. The reproducer is already in our tree, it
   is pure C with no wasm involved, and the fix is a single line we can revert --
   so the control is exact rather than approximate.
2. **Entry 6**, CVE-2026-54913. Unpatched, specific to `WASM_ENABLE_GC=0` which is
   our build, and the module is 90 bytes.
3. **Entry 13**, the memsys5 free-list over-read, but only after the rounding caveat
   above has been negative-tested. Otherwise the result is meaningless either way.
4. **Entry 28**, Contiki-NG `IN_HEAP`. Not a subject we run, but it is the cleanest
   write-anywhere-from-a-foreign-pointer specimen in the whole set and would port to
   a shim in an afternoon.
