# MPY-T28: gc_realloc returns NULL after a few calls
Source: #322, https://github.com/micropython/micropython/issues/322  
Upstream state: closed, first seen 2014-02-25

**BLOCKED on a C-level harness. No Python trigger exists.**

## The defect

gc_realloc returns NULL after a few calls, caller keeps using the old block.

Class `alloc-invariant`, CWE-476, in `py/gc.c:gc_realloc`. Scope `gc-core`, so it lives on memory MicroPython's own collector manages,
inside the single region `gc_init` was handed.

## What Capstone does about it

`traps_unmodified` = **no**. An unmodified runtime gets no temporal protection here: the heap is one
object, every sub-allocation inherits its bounds, and `gc_free` never
reaches the hardware, so there is nothing to revoke. See
`../../evidence/heap-bounds-model.s` and
`../../evidence/nested-uaf-qemu-2026-08-17.txt`.

The same blindness is not special to capabilities. AddressSanitizer misses
this defect family too, in a toolchain where it catches an ordinary `malloc`
use-after-free, because the runtime's frees never reach it either:
`../../evidence/asan-blindness-2026-08-17.txt`.

`traps_if_gc_cap_aware` = unclear, which is a prediction about a capability-aware
collector that does not exist yet. Not evidence.

## Measured

Not run. The published repro calls gc_alloc/gc_realloc from c; no python trigger exists.

## Reproducing

Not reproducible with the current setup: the published repro calls gc_alloc/gc_realloc from C; no Python trigger exists.
