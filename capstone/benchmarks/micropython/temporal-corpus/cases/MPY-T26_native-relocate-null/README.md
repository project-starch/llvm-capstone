# MPY-T26: Segmentation Fault (NULL Pointer Dereference) in mp_native_relocate
Source: #18645, https://github.com/micropython/micropython/issues/18645  
Upstream state: closed, first seen 2026-01-06

**BLOCKED. Upstream status unresolved, see below.**

## The defect

relocation walks a pointer table that is NULL or already freed.

**NOT a temporal defect**, and kept only as a labelled counter-example:
a malformed .mpy drives relocation off a NULL table; input validation. It was classified from its title before the fix
commit was read; the 2026-08-18 audit corrected it.

Class `dangling-pointer`, CWE-476, in `py/nativeglue.c:mp_native_relocate`. Scope `gc-managed`, so it lives on memory MicroPython's own collector manages,
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

Not run. Needs the native emitter and a relocatable .mpy.

## Reproducing

Not reproducible with the current setup: needs the native emitter and a relocatable .mpy.
