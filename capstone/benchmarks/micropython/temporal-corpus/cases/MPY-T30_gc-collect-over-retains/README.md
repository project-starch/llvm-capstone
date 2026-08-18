# MPY-T30: gc.collect() does not work properly
Source: #11698, https://github.com/micropython/micropython/issues/11698  
Upstream state: open, first seen 2023-06-04

**NOT REPRODUCIBLE HERE. The trigger cannot be expressed in this domain.**

## The defect

gc.collect() does not reclaim what the caller expects, blocks stay marked live.

**NOT a temporal defect**, and kept only as a labelled counter-example:
gc.collect retaining more than expected is over-retention, the opposite of premature free. It was classified from its title before the fix
commit was read; the 2026-08-18 audit corrected it.

Class `alloc-invariant`, CWE-401, in `py/gc.c:gc_collect`. Scope `gc-core`, so it lives on memory MicroPython's own collector manages,
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

Not run. Reported against the esp32 port and closed needs-info upstream.

## Reproducing

Not reproducible with the current setup: reported against the ESP32 port and closed needs-info upstream.
