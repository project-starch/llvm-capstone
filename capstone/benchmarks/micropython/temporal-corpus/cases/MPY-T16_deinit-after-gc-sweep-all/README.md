# MPY-T16: MICROPY_PORT_DEINIT_FUNC called after gc_sweep_all
Source: #5487, https://github.com/micropython/micropython/issues/5487  
Upstream state: open, first seen 2020-01-03

**NOT REPRODUCIBLE HERE. The trigger cannot be expressed in this domain.**

## The defect

port deinit hook runs after the sweep that already freed what it touches.

**Temporal: yes.** port deinit runs after the sweep that already freed what it touches.

Class `lifetime-order`, CWE-416, in `py/gc.c:gc_sweep_all`. Scope `gc-core`, so it lives on memory MicroPython's own collector manages,
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

Not run. Needs a port shutdown hook; this domain's teardown is not the one with the defect.

## Reproducing

Not reproducible with the current setup: needs a port shutdown hook; this domain's teardown is not the one with the defect.
