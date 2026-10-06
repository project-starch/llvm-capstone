# Provenance

**Upstream fix:** `gh-157335`, *"[3.13] gh-157335: Fix out-of-bounds write in mmap.mmap.__setitem__ (#157438)"*.
**Consumer:** `mmapmodule.c`. **CVE:** `NO VERIFIED CVE`.

## Liveness at the pin, measured

measured, not asserted: the trigger runs on the pinned 3.13.7 ASan build (PYTHONMALLOC=malloc, ASAN_OPTIONS=detect_leaks=0) and reports ASAN:SEGV in safe_byte_copy (mmapmodule.c:361). Running it is the liveness proof -- a sanitizer report here means the defect is present in this tree.

Running the trigger IS the liveness proof. A sanitizer report on this tree means
the defect is in this tree, so nothing here rests on a commit message, a CPE or
a version range.

Two conditions had to hold for that run to mean anything, both learned the hard
way on this corpus:

* `ASAN_OPTIONS=detect_leaks=0`. With leak checking on, the leak summary buries
  the report and a reproducing case reads as silent.
* A positive control before the batch. An earlier run capped address space with
  `ulimit -v`, which stops ASan reserving its shadow region
  (`ReserveShadowMemoryRange failed`); 84 cases came back "no violation" when
  the instrument had never started.

## Which allocator owns the object

**non-nested: the memory comes straight from libc malloc** -- `allocator_layer` = `libc malloc (the object never reaches a pymalloc pool)`, `allocator_consumed` = `no`.

NOT measured as a region -- the buffer IS an mmap mapping (mmapmodule.c), so a write past its end reaches the OS and ASan reports SEGV rather than a heap region. No pymalloc pool is involved in any configuration

This was decided by running the trigger twice, not by reading the region size.
Under `PYTHONMALLOC=malloc` every allocation is one libc malloc, so ASan reports
a region for everything and the size cannot tell the domains apart -- case 07
overflows a 100-byte region and is still non-nested, because
`PyOS_StdioReadline`'s buffer is `PyMem_RawMalloc`'d. The question that does
separate them is whether the heap report **survives**
`PYTHONMALLOC=pymalloc`.

## Fidelity

**interpreter-trigger / real allocator.** The trigger is upstream's own
reproducer driven through the real CPython, not a C reduction against the
allocator the way `../pymalloc-repros` does it. That is the stronger end for
liveness -- the defect is reached in place, in the module it lives in -- and it
says nothing about which arm ought to catch it.

## Arms

None run. All four are declared `{"status": "not written"}` in `case.json` so
the gap is visible rather than silently absent.
