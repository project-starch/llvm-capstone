# MPY-T03: A flaw has been found in micropython up to 1.27.0. This vulnerability affects the function mp_import_all of the file py/runtime.c. This manipulation c
Source: CVE-2026-1998, https://nvd.nist.gov/vuln/detail/CVE-2026-1998  
Upstream state: patched, first seen 2026-02-06

**BLOCKED on a parent build. Already fixed in the pinned source.**

## The defect

import * over a module whose globals map is mutated during iteration.

Class `memory-corruption`, CWE-119,CWE-787, in `py/runtime.c:mp_import_all`. Scope `gc-managed`, so it lives on memory MicroPython's own collector manages,
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

Not run. Fixed at the pin.

## Reproducing

The defect is fixed in the pinned source, so building the pin measures nothing.
Build `570744d06c5b^` instead, the fix commit's parent:

```bash
MPY_COMMIT=570744d06c5b^ bash ../../../fetch-micropython.sh
```

**This was attempted and it does not currently work**, and the blocker is not
the one originally predicted. A two-year-old MicroPython does not build
usefully with a current toolchain: GCC 15 rejects the tree, `mpy-cross`
crashes freezing its own manifest, and the minimal variant segfaults on a
plain `bytearray`. Full account, with the exact errors, in
`../../evidence/parent-build-attempt-2026-08-17.txt`. Retrying this needs a
period-correct compiler, most likely in a container.
