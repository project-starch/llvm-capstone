# MPY-T19: BLE: crash after running gc.collect()
Source: #5226, https://github.com/micropython/micropython/issues/5226  
Upstream state: closed, first seen 2019-10-18

**NOT REPRODUCIBLE HERE. The trigger cannot be expressed in this domain.**

## The defect

gc.collect() collects a BLE object still referenced only from C.

Class `premature-free`, CWE-416, in `extmod/modbluetooth.c`. Scope `gc-managed`, so it lives on memory MicroPython's own collector manages,
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

Not run. The fix touches extmod/modbluetooth_nimble.c; reproducing needs a nimble stack, which no host build provides.

## Reproducing

Not reproducible with the current setup: the fix touches extmod/modbluetooth_nimble.c; reproducing needs a NimBLE stack, which no host build provides.
