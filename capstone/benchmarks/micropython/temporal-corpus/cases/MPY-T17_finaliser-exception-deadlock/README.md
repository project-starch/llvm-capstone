# MPY-T17: raising exception in __del__ finaliser results in deadlock with multithread enabled
Source: #3627, https://github.com/micropython/micropython/issues/3627  
Upstream state: open, first seen 2018-02-21

**NOT REPRODUCIBLE HERE. The trigger cannot be expressed in this domain.**

## The defect

exception raised inside a __del__ finaliser deadlocks with threading on.

Class `lifetime-order`, CWE-667, in `py/gc.c,py/objtype.c`. Scope `gc-core`, so it lives on memory MicroPython's own collector manages,
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

`traps_if_gc_cap_aware` = not-trapped, which is a prediction about a capability-aware
collector that does not exist yet. Not evidence.

## Measured

Not run. Needs micropy_py_thread, which is off here.

## Reproducing

Not reproducible with the current setup: needs MICROPY_PY_THREAD, which is off here.
