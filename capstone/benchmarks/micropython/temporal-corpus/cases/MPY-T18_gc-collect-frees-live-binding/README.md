# MPY-T18: Crash in LVGL lv_draw_dispatch_layer() after gc.collect(), layer->draw_task_head corrupt
Source: #19413, https://github.com/micropython/micropython/issues/19413  
Upstream state: closed, first seen 2026-07-03

**BLOCKED. Upstream status unresolved, see below.**

## The defect

gc.collect() frees a structure the C binding still holds, draw_task_head corrupt.

Class `premature-free`, CWE-416, in `py/gc.c + LVGL binding`. Scope `gc-managed`, so it lives on memory MicroPython's own collector manages,
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

Not run. Needs the lvgl c binding; not part of this port.

## Reproducing

Not reproducible with the current setup: needs the LVGL C binding; not part of this port.
