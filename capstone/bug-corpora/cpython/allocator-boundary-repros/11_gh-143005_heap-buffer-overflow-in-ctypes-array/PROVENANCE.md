# Provenance

**Upstream fix:** `gh-143005`, *"Heap buffer overflow in ctypes array assignment via `__class__` swap"*.
**Consumer:** `unknown.c`. **CVE:** `NO VERIFIED CVE`.

## Liveness at the pin, measured

measured, not asserted: the trigger runs on the pinned 3.13.7 ASan build (PYTHONMALLOC=malloc, ASAN_OPTIONS=detect_leaks=0) and reports ASAN:heap-buffer-overflow in a 24-byte region. Running it is the liveness proof -- a sanitizer report here means the defect is present in this tree.

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

**nested: the memory comes from a pymalloc pool** -- `allocator_layer` = `pymalloc`, `allocator_consumed` = `yes`.

(no layer note recorded)

This was decided by running the trigger twice, not by reading the region size.
Under `PYTHONMALLOC=malloc` every allocation is one libc malloc, so ASan reports
a region for everything and the size cannot tell the domains apart -- case 07
overflows a 100-byte region and is still non-nested, because
`PyOS_StdioReadline`'s buffer is `PyMem_RawMalloc`'d. The question that does
separate them is whether the heap report **survives**
`PYTHONMALLOC=pymalloc`.

A crash without a heap report in the pymalloc arm is not a missed
measurement: the stale read returns the freelist link pymalloc writes
INSIDE the freed block (`Objects/obmalloc.c:2494`), so the program dies
on that garbage while ASan stays quiet about the heap. That is the
mechanism evidence for "consumed inside the arena".

## Fidelity

**interpreter-trigger / real allocator.** The trigger is upstream's own
reproducer driven through the real CPython, not a C reduction against the
allocator the way `../pymalloc-repros` does it. That is the stronger end for
liveness -- the defect is reached in place, in the module it lives in -- and it
says nothing about which arm ought to catch it.

## Arms

None run. All four are declared `{"status": "not written"}` in `case.json` so
the gap is visible rather than silently absent.
