# perlbench test: which part of the allocator's excess over Plain is avoidable (pre-registration)

Branch `perf/heap-excess`, on dev a6956c05a6d1. Question: in 500.perlbench_r test, Capstone's mallocng and
copy routines run +57.4 (allocator) and +10.1 (memmove) instructions per 1,000 Plain instructions more than
Plain's. Which part can go without weakening the heap's spatial or lifetime guarantees?

## Instrument

Per-PC capmix3 (shift 2), in-context windows (mark1/mark2), summed over test.pl's two commands:
- Capstone: the compiler lane's profile of the udcnofp image (exe 5996ba25b57a303c, kit-unq). The exe runs at
  28 bases (one per context); each base is chosen by where the most `ldc` executions land, and a PC is kept only
  where the plugin's opcode class agrees with the disassembly. 90.66 G of the 95.79 G user-range instructions in
  the windows map to the exe (the rest is the launcher and natperl). In-context instret of that run: 3,458,228,002
  + 87,663,678,223 = 91,121,906,225.
- Plain: perl-plainr (plain-rv64r, non-PIE), 80,327,966,980 instructions in the windows. All rates below are
  per 1,000 of these.

The per-function numbers reproduce the compiler lane's table to two decimals.

## Attribution (per 1k Plain; Capstone vs Plain; quoted per-PC counts)

| line | Cap | Plain | cause | class |
|---|---:|---:|---|---|
| malloc body | 35.28 | 16.73 | `__libc_malloc_impl` IS patch 0009's new `malloc` (glue.h `#define malloc __libc_malloc_impl`); the public `malloc` is a 4-instruction stub to it (17.7 M calls). 19.98 M calls; 278,019 (1.4 %) take the `n >= 4096` aligned_alloc path, whose `heap_granule` is a software CLZ (no Zbb). `heap_issue` per allocation | issue REQUIRED; stub, frame, CLZ CODEGEN |
| `__libc_free` | 31.81 | 17.75 | `heap_access`: `lbu zero,0(p)` liveness probe, cursor>>39 and base==cursor checks, a three-level page-index walk to the parent: ~33 instructions per call, 18.2 M calls. `heap_end` (1 instruction). A 10-instruction prologue taken before the `if (!p)` test | probe, checks, walk, end REQUIRED; frame CODEGEN |
| get_meta | 15.17 | 10.76 | not inlined at -O1 (18.3 M calls, 66 per call); every global access is auipc/addi/scc/mv/add/shrink | CODEGEN |
| `__malloc_allzerop` | 9.91 | 4.35 | calloc asks it of the child malloc just issued, so it runs `heap_access` again (`lbu zero,0x0(a0)` at the entry, 5,774,279 executions) to find the pointer malloc already had | REDUNDANT |
| `__libc_realloc` | 7.77 | 4.18 | `heap_round`, `heap_access`, granule checks, issue/end on in-place resize; the moving path's `free(issued)` repeats `heap_access` (2,237,124 times, ~0.9 per 1k) | REQUIRED; the repeat REDUNDANT only without a racing free, NOT taken |
| calloc | 4.57 | 2.97 | wrapper chain and -O1 | CODEGEN |
| `__unlock` | 4.06 | 0.95 | NOT locking: `__lock` runs 0 times in the windows, and `__libc_free`'s `if (!MT)` branch is taken 13,100,426 of 13,100,426 times, so `need_locks` is 0. `__unlock`'s `bgez a0` (lock word not negative) is taken 25,093,567 of 25,093,567 times; the 13 instructions per call are the frame built before that test (-O1 and -O2 alike: the backend does not shrink-wrap) | REDUNDANT call, CODEGEN cause |
| alloc_slot, enframe, nontrivial_free | 31.28 | 28.15 | same mallocng code; capability arithmetic and -O1 | CODEGEN |
| memmove | 12.34 | 2.20 | Capstone's memmove DOES forward non-overlapping copies to the memcpy body (`copy_forward`); Plain's forwards to memcpy, so Plain's cost sits under memcpy (Plain memcpy 19.02 vs Capstone 17.30). The gap is the one-byte tail loop: 78,690,250 iterations of 7 instructions (551 M of memmove's 991 M), ~9 bytes per call, 8,927,838 calls whose addresses differ mod 4 going straight to it | REDUNDANT (no tag or bounds reason) |

Tag-safety of a word copy (the memmove question). Every scalar store clears the tag of each granule it touches,
whatever its width (capstone-qemu `helper_remove_cap_mem_map`: `cap_mem_map_remove_range(&env->cm_map, addr,
size)`), and only a capability store sets one. An aligned 8-byte store inside [dst, dst+n) touches exactly the
granules the byte loop touches over the same bytes, so it can neither forge nor keep a tag the bytes would not.
This is the argument the file already makes for its mod-8 and mod-4 word paths (c7f689604bc4). R-29 concerns
128-bit loads only; plain `ld` is always correct there.

## Change (this branch)

1. musl 0011: calloc's small path checks for a zero slot through mallocng's own pointer, then issues the child
   exactly as malloc does; and `unlock()` makes `__unlock`'s own lock-word test before the call.
2. libc at -O2 (survey-musl-capstone.py), as musl's own build and Plain's.
3. runtime string_bounds_safe.c: the byte tail eight at a time, then 4/2/1, at constant offsets; a shift-merge
   8-byte word path for copies of 32 bytes or more whose addresses differ mod 8.

Not changed, REQUIRED: heap_access in free and realloc, heap_issue, heap_end, representable rounding, the
mallocng consistency checks.

## Pre-registered prediction (written before either perl run)

Arms, both relinked from the udcnofp objects (control relink reproduces 5996ba25b57a303c exactly), profiled the
same way (kit-unq, marked stage, CAPMIX_SHIFT=2):
- o2: libc -O2 only. In-context instret -0.2 to -0.9 G against 91,121,906,225 (the gcc/PHP -O2 experiment gave
  -0.50 to -0.62 %).
- x: o2 plus 0011 and the string change. Against o2: calloc -0.15 to -0.30 G (heap_access 33 x 5.77 M plus the
  stub and wrapper), unlock -0.20 to -0.28 G (25.1 M x ~10), memmove/memcpy tail -0.2 to -0.4 G. x against the
  baseline: -0.75 to -1.9 G, i.e. -9 to -24 per 1k Plain. Per function: `__malloc_allzerop` to ~0 (only n >= 4096),
  `__unlock` to under 0.5 per 1k, memmove under 8 per 1k.
- Both: rc 0, specdiff 2/2 PASS, every output hash equal to the baseline run's except files that carry the
  launcher's timing line.
- Safety: run-malloc.py and run-hosted.py on the control SDK and the x SDK give the same verdict per case; a
  mutant libc whose free does not end the child's lifetime fails the lifetime cases; a mutant runtime that moves
  granules as words fails memmove_ranges_tags / memcpy_ranges_tags.

If x saves less than 10 per 1k Plain, the avoidable part is small and that is the answer.
