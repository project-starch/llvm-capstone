# MPY-T29: The assert in gc_free(...) and gc_realloc(...) will fail
Source: #4705, https://github.com/micropython/micropython/issues/4705  
Upstream state: closed, first seen 2019-04-19

**NOT REPRODUCIBLE HERE. The trigger cannot be expressed in this domain.**

## The defect

assertions in gc_free and gc_realloc fail on a pointer they consider invalid.

Class `alloc-invariant`, CWE-617, in `py/gc.c:gc_free,gc_realloc`. Scope `gc-core`, so it lives on memory MicroPython's own collector manages,
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

`traps_if_gc_cap_aware` = trapped, which is a prediction about a capability-aware
collector that does not exist yet. Not evidence.

## Measured

Not run. The fix is ports/unix/gccollect.c, making the gc capture stack and registers properly; the trigger depends on what happens to be in registers and is not deterministically reproducible.

## Reproducing

Not reproducible with the current setup: the fix is ports/unix/gccollect.c, making the GC capture stack and registers properly; the trigger depends on what happens to be in registers and is not deterministically reproducible.
