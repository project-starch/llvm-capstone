# MPY-T06: heap-use-after-free when reusing a closed btree object
Source: #12543, https://github.com/micropython/micropython/issues/12543  
Upstream state: closed, first seen 2023-09-28

**BLOCKED on a parent build. Already fixed in the pinned source.**

## The defect

btree object used again after close() released its backing store.

Class `uaf`, CWE-416, in `extmod/modbtree.c`. Scope `gc-managed`, so it lives on memory MicroPython's own collector manages,
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

Not run. Fixed at the pin; also needs the btree module, which this domain does not build.

## Reproducing

The defect is fixed in the pinned source, so building the pin measures nothing.
Build `6db91dfefb1a^` instead, the fix commit's parent:

```bash
MPY_COMMIT=6db91dfefb1a^ bash ../../../fetch-micropython.sh
```

**This was attempted and it does not currently work**, and the blocker is not
the one originally predicted. A two-year-old MicroPython does not build
usefully with a current toolchain: GCC 15 rejects the tree, `mpy-cross`
crashes freezing its own manifest, and the minimal variant segfaults on a
plain `bytearray`. Full account, with the exact errors, in
`../../evidence/parent-build-attempt-2026-08-17.txt`. Retrying this needs a
period-correct compiler, most likely in a container.
