# MPY-T07: bug: lex->source_name used after lex is freed
Source: #4128, https://github.com/micropython/micropython/issues/4128  
Upstream state: closed, first seen 2018-09-12

**BLOCKED. Upstream status unresolved, see below.**

## The defect

lex->source_name read after the lexer allocation was freed.

Class `uaf`, CWE-416, in `py/lexer.c,py/compile.c`. Scope `gc-managed`, so it lives on memory MicroPython's own collector manages,
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

Not run. No fix commit names the issue and no python trigger is published.

## Reproducing

Not reproducible with the current setup: no fix commit names the issue and no Python trigger is published.
