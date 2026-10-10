# CPython allocator-boundary defect corpus

21 memory-safety defects live in CPython 3.13.7, in the three cells
[`pymalloc-repros`](../pymalloc-repros) does not cover. That corpus holds
consumer-side **temporal** defects whose memory comes from **pymalloc** and
states it is the whole reachable set at the pin; these are the others.

Together the two corpora are 41 cases and all four cells:

| | nested (pymalloc) | non-nested (libc malloc) | total |
|---|---:|---:|---:|
| **temporal** | 20 — `pymalloc-repros` | **3** — here | 23 |
| **spatial** | **13** — here | **5** — here | 18 |
| **total** | **33** | **8** | **41** |

## What nested and non-nested mean here

CPython takes a 1 MiB arena from the OS, cuts it into 16 KiB pools and hands out
blocks of 512 bytes or less (`SMALL_REQUEST_THRESHOLD`,
`Include/internal/pycore_obmalloc.h:157`). A use-after-free or an overflow
between two of those blocks never crosses the boundary of that one `mmap`, and
`free()` never reaches `malloc`, so a malloc-level tool has no event to see.
That is **nested**. **Non-nested** means the object came straight from libc
`malloc`, with no second allocator in between.

Two routing facts decide which applies, both read off the pinned source:

* `Objects/obmalloc.c:351` sets `PYMEM_ALLOC = PYMALLOC_ALLOC`, so the MEM
  domain is **not** a separate allocator. Only `PyMem_RawMalloc` bypasses
  pymalloc.
* Requests over 512 bytes go to libc `malloc` even through the OBJ and MEM
  domains.

## The side is measured, not read off the size

The obvious shortcut -- "ASan reported an N-byte region, N ≤ 512, so it is
nested" -- is wrong, and case 07 is why. `gh-140594` overflows a **100-byte**
region and is non-nested: `PyOS_StdioReadline`'s buffer is `PyMem_RawMalloc`'d
(`Parser/myreadline.c:207,223,325,347`), so libc owns it at any size. Under
`PYTHONMALLOC=malloc` every allocation is a libc malloc, so ASan reports a
region for everything and the size cannot tell the domains apart.

So each trigger is run **twice** and the question is whether the heap report
survives the nested allocator:

| under `PYTHONMALLOC=pymalloc` | verdict |
|---|---|
| ASan still reports a **heap** violation | the object is not a pymalloc block → **non-nested** |
| no heap report | the damage stayed inside the arena → **nested** |

A crash without a heap report in the pymalloc arm is **not** a missed
measurement. The stale read returns the freelist link pymalloc writes *inside*
the freed block (`Objects/obmalloc.c:2494`), so the program dies on that
garbage while ASan stays quiet about the heap. That is the mechanism evidence
for "consumed inside the arena", and it is what `allocator_consumed` records.

Both halves of axis 2 are present on every case: `allocator_layer` says which
allocator the memory came from, `allocator_consumed` whether the damage stayed
inside one of its blocks.

## Liveness

Every case is live at the pin and the proof is a measurement, never an
assertion: the trigger is upstream's own reproducer from the issue, and running
it on the pinned 3.13.7 ASan build produces the sanitizer report quoted in
`live_proof`. All 32 reproduce from their case directory with the oracle
recorded for them -- the 21 filling the three cells `pymalloc-repros`
does not cover, and the 11 temporal/nested cases that re-measure its
issues through the interpreter.

Three things that had to be got right for those runs to mean anything:

* `ASAN_OPTIONS=detect_leaks=0`. With leak checking on, the leak summary buries
  the report and a reproducing case reads as silent.
* A positive control before each batch. An earlier run of this corpus capped
  address space with `ulimit -v`, which stops ASan reserving its ~15 TB shadow
  region; 84 cases came back "no violation" when the instrument had never
  started.
* A negative control, because a positive one only proves the instrument
  started. Two exist and they answer different questions: a weak one that
  replaces every trigger with a stub, which the three arms of 2026-10-06 (the
  physical `spatial` and `sublet`, and `cheribsd-revocation`) pass, and a strong
  per-case one that keeps the allocation traffic and makes only the offending
  access valid, which exists for 18 of the 32 cases and has been run on the
  base arm. For the remaining 14 the defect is not one invalid access in an
  otherwise ordinary program, and three of those 14 are the cases no arm
  delivers. What each control settles, and what neither does, is in
  `results/20261006/README.md`. Neither has been run on the virtual arms yet:
  `runners/virtual/run-virtual.py` has no negative-control mode.

## Arms

CPython has one heap on Capstone, musl's mallocng on the virtual profile
(`capstone/runtime/virtual/heap.c`): exact bounds per object, lifetime retired on free. The two
Capstone arms differ only in pymalloc, and the arm names are the ones every corpus on the board
uses:

| arm | configuration (`tools/arms.json`) | what it is |
|---|---|---|
| `virtual-malloc` | `virtual-cpython` | the interpreter with pymalloc stock (patch 0009). mallocng bounds and retires what it hands out -- `PyMem_Raw` memory, objects over 512 bytes, pymalloc's arenas -- but not the blocks pymalloc carves inside an arena |
| `virtual-nested-pools` | `virtual-cpython-pools` | the same with `CPY_SUBLET=1`: patch 0014 makes obmalloc hand out every block as a child lifetime of its arena, bounded to the request (`CDERIVE`), and revoke it on free (`CREVOKE`). Requests over 512 bytes stay with mallocng, as in the stock arm |
| `cheribsd-revocation` | -- | CheriBSD purecap, libc revocation as the platform ships it |

How to build and run the two Capstone arms is in
[`runners/virtual/README.md`](runners/virtual/README.md). A fault counts for a case only when it
lies in one of the case's `fault_sites`, which come from a host ASan run recorded before any arm
ran (`probe/asan-sites.py`, `results/2026-10-10-asan/sites.tsv`). That run locates 31 of the 32
cases; case 17 produced no report there, although the 2026-10-07 ASan build recorded one, so a
Capstone fault in case 17 cannot be attributed.

**Superseded: the physical arms.** Until 2026-10-10 the Capstone arms ran on the physical
application domain, `spatial` (the level0 heap with per-object bounds, pymalloc stock) and `sublet`
(level0 plus patch 0014). CPython no longer builds against those heaps. Their results, the
expectation pre-registered for them and how it held up are in
[`results/20261006/README.md`](results/20261006/README.md); `probe/counts.py --check` still
re-derives that page's tables from its matrix.

## Fidelity, stated as a limitation

The triggers are upstream reproducers driven through the real interpreter, not
C reductions against the allocator the way `pymalloc-repros` does it. They
establish that the defect is live and which allocator owns the object. They do
**not** establish which arm ought to catch it.
