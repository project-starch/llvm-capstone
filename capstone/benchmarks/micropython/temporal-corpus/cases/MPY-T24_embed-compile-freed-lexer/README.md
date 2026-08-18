# MPY-T24: ports/embed: segfault in mp_compile()
Source: #11781, https://github.com/micropython/micropython/issues/11781  
Upstream state: closed, first seen 2023-06-14

**MEASURED at the fix commit's parent. Fixed in the pinned source.**

## The defect

mp_compile() on an embed port faults on a freed lexer or parse tree.

**NOT a temporal defect**, and kept only as a labelled counter-example:
fix: 'embed: Improve stack top estimation'; a stack overflow, no lifetime component. It was classified from its title before the fix
commit was read; the 2026-08-18 audit corrected it.

Class `uaf`, CWE-416, in `ports/embed, py/compile.c`. Scope `gc-managed`, so it lives on memory MicroPython's own collector manages,
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

- stock MicroPython at the pin: **not-reproducible**
- Capstone domain under QEMU: **not-run**

See `RESULT.txt` in this directory.

## Reproducing

The defect is fixed in the pinned source, so building the pin measures nothing.
Build `d2a3cd7ac428^` instead, the fix commit's parent:

```bash
MPY_COMMIT=d2a3cd7ac428^ bash ../../../fetch-micropython.sh
```

**This works.** Use gcc-12, a compiler contemporary with the commit; the
default gcc 15 rejects the tree. Full recipe, and the two non-obvious
flags it needs, in `../../evidence/parent-build-attempt-2026-08-17.txt`.
Do NOT add AddressSanitizer: it breaks the MicroPython unix port outright,
and the question it would have answered is already settled in
`../../evidence/asan-blindness-2026-08-17.txt`.
