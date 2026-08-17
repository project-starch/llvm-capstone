# MPY-T14: crash with inappropriate readblocks implementation
Source: #19060, https://github.com/micropython/micropython/issues/19060  
Upstream state: closed, first seen 2026-04-08

**BLOCKED on a parent build. Already fixed in the pinned source.**

## The defect

readblocks implementation enlarges the buffer it was handed.

Class `dangling-buffer`, CWE-416, in `extmod/vfs_blockdev.c`. Scope `gc-managed`, so it lives on memory MicroPython's own collector manages,
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

Not run. Fixed at the pin; duplicate of mpy-t15.

## Reproducing

The defect is fixed in the pinned source, so building the pin measures nothing.
Build `64f0394d80ca^` instead, the fix commit's parent:

```bash
MPY_COMMIT=64f0394d80ca^ bash ../../../fetch-micropython.sh
```

**This works.** Use gcc-12, a compiler contemporary with the commit; the
default gcc 15 rejects the tree. Full recipe, and the two non-obvious
flags it needs, in `../../evidence/parent-build-attempt-2026-08-17.txt`.
Do NOT add AddressSanitizer: it breaks the MicroPython unix port outright,
and the question it would have answered is already settled in
`../../evidence/asan-blindness-2026-08-17.txt`.
