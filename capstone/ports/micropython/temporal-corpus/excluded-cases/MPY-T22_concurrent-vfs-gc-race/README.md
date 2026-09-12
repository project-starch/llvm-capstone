# MPY-T22: multi-thread file access causing system crash
Source: #17442, https://github.com/micropython/micropython/issues/17442  
Upstream state: open, first seen 2025-06-06

**NOT REPRODUCIBLE HERE. The trigger cannot be expressed in this domain.**

## The defect

concurrent file access, allocator state mutated from two threads.

**Temporal: uncertain.** a data race on allocator state; whether it manifests as a lifetime violation is not established.

Class `race-uaf`, CWE-362, in `py/gc.c + vfs`. Scope `gc-core`, so it lives on memory MicroPython's own collector manages,
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

Not run. Needs threads and a filesystem, both absent here.

## Reproducing

Not reproducible with the current setup: needs threads and a filesystem, both absent here.
