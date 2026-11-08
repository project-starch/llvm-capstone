# perlbench test: which part of the allocator's excess over Plain is avoidable

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

## Results (after the runs; pre-registration commit 76455aff7e7a)

In-context instret, test.pl's two commands summed, same runner, kit (kit-unq), marked stage and CAPMIX_SHIFT=2 as
the baseline profile:

| arm | exe | instret | vs baseline | per 1k Plain |
|---|---|---:|---:|---:|
| baseline (udcnofp profile) | 5996ba25b57a303c | 91,121,906,225 | | |
| o2: libc -O2 | f009a98694ef2cbc | 90,733,128,002 | -388,778,223 (-0.43 %) | -4.84 |
| x: o2 + 0011 + string change | cfb5ce3d66ae7ec0 | 89,962,022,430 | -1,159,883,795 (-1.27 %) | -14.44 |

x against o2: -771,105,572, -9.60 per 1k. Both arms: rc 0, specdiff 2/2 PASS, 558 of 560 output hashes equal to
the baseline run's; the two that differ, test.err and makerand.err, hold only the launcher's per-context
CAPSTONE_VM_STATS/STEPS lines (same 632 and 2 lines, differing in their numbers). The same-exe noise between the
compiler lane's capmix and plain runs of udcnofp is 1.16 M.

Per function (per 1k Plain; baseline / o2 / x; Plain): `__unlock` 4.06 / 4.06 / 0.00 (0.95); `__malloc_allzerop`
9.91 / 9.19 / 0.00, replaced in x by `slot_allzero` 6.75 and `__malloc_calloc_alloc` 2.55 (4.35);
`__libc_free` 31.81 / 29.51 / 29.25 (17.75); `__malloc_alloc` 26.45 / 25.93 / 25.18; memmove 12.34 / 12.34 / 9.59
(2.20); memcpy 17.30 / 17.31 / 16.83 (19.02). Mapped exe totals 1133.9 / 1129.3 / 1119.5.

Against the prediction:
- o2 -0.39 G: inside -0.2 to -0.9 G. Most of it is the allocator (free -2.30, allzerop -0.72, `__malloc_alloc`
  -0.52 per 1k). -O2 does not shrink-wrap: `__unlock` and free's 10-instruction entry are unchanged.
- unlock: -5.1 per 1k (`__unlock` -4.06, and `__malloc_alloc`/free shrink by 0.75/0.26 where the call became a
  test). ABOVE the band of -2.5 to -3.5: the band did not count the callers' own savings.
- calloc: -1.28 per 1k, BELOW the band of -1.9 to -3.7. The lookup did go (allzerop 9.19 -> slot_allzero 6.75,
  34 instructions per call), but `__malloc_calloc_alloc` costs what the malloc wrapper and stub it replaces cost
  (+2.55 against -2.11), and calloc itself grew by 0.72 (the zero flag goes through memory). The band assumed the
  wrapper would go too.
- memmove + memcpy: -3.23 per 1k, inside -2.5 to -5. The 8/4/2/1 tail carries it; the shift path ran in 364,035
  of the 11.1 M calls whose addresses differ mod 8 (2.35 M words).
- x -14.44 per 1k: inside -9 to -24.

Of the +57.4 allocator excess, -O2 and 0011 remove 11.1 (-4.8 build, -6.4 source); the copy routines remove 3.2 of
memmove's. What remains is REQUIRED (heap_access in free and realloc, issue, end) or CODEGEN: no shrink-wrapping,
a frame pointer and 4-instruction call stubs in libc (the application is built with -fomit-frame-pointer and
-capstone-direct-calls, libc is not), six instructions per global access, a software CLZ, and two capability moves
per byte-loop iteration.

## Safety fixtures (kit-unq, domain slot 76)

| runner | control SDK (devrg) | x SDK | positive control |
|---|---|---|---|
| run-hosted.py (94 checks, incl. memcpy/memmove_ranges_tags: every offset mod 16/32, len <= 80, both overlaps, exact bounds) | PASS 94/94 | PASS 94/94 | runtime whose granule copy is two words: 82/94, FAIL memcpy_ranges_tags, memmove_ranges_tags and 10 other pointer-copy and exit checks |
| run-malloc.py (24 checks: stale, stale-free, inplace-stale, remap-stale, bounds, interior-free, reuse-stale, reuse-free, nested-*) | PASS 24/24 | PASS 24/24 | libc whose free does not end the child: FAIL stale, stale-free, inplace-stale, reuse-stale, reuse-free ("denial survived") |

All 11 cases that fault on the control fault on x, each with the same exit (139) and the same cause after its READY marker: 25 (revoked capability) for the eight lifetime cases, 28 for bounds, 2 for interior-free, 29 for nested-grandchild. The
devrg manifest predates run-malloc.py's `heap_object_bounds` key; the runs used a copy with the key set true, which
is what the archive was built with (the linked policy symbol is `__malloc_policy_11`).

Host-side, before any run: the new copy matched glibc memmove on 2,469,888 (src offset, dst offset, length,
overlap) cases, and a variant with a wrong shift failed 218,432 of them; under ASan with every byte outside
[src, src+n) and [dst, dst+n) poisoned the copy read and wrote nothing outside, and shrinking the source by one byte
made ASan fire.

## Not done

- realloc's moving path repeats `heap_access` in `free(issued)` (2,237,124 times, ~0.9 per 1k). Skipping it would
  also skip the liveness probe that catches a free racing the realloc; left in.
- libc-test at -O2 was not run; the perl run and the two fixture suites are the -O2 evidence.
