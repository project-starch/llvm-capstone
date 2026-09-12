# MPY-T12: Null pointer dereference in dict equality after re-entrant `__bool__` clears RHS map
Source: #18619, https://github.com/micropython/micropython/issues/18619  
Upstream state: open, first seen 2025-12-30

**MEASURED in the domain.**

## The defect

re-entrant __bool__ clears the RHS map during dict equality, map pointer goes NULL.

**Temporal: yes.** reporter: 'the lookup returns a slot pointer into the now-freed table'.

Class `reentrancy`, CWE-476, in `py/objdict.c`. Scope `gc-managed`, so it lives on memory MicroPython's own collector manages,
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

- stock MicroPython at the pin: **crash-sigsegv**
- Capstone domain under QEMU: **fault-cause24**

See `RESULT.txt` in this directory.

## Reproducing

`repro.py` is the script. On stock:

```bash
MPY=/tmp/capstone/mpy-stock-pin/ports/unix/build-standard/micropython
$MPY repro.py
```

In the domain, copy it into the test directory the image is built from and
follow `../../README.md`; the driver is `tools/run-resumable-suite.py`
and it must be run with `--capture-output`, because a test that dies on a
missing builtin still returns a retval and reads exactly like an untrapped one.
