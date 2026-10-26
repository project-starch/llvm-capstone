# CRASH-081 and CRASH-075: the two cases the 29-case run could not read

Two of the three cells the `temporal-asan-visible` run refused were `setup-fault` on every arm
(CRASH-068 is the third and is not addressed here). Both turned out to be **resource ceilings of
the port and the platform, not protection failures**, and one of them is now a catch.

Nothing in `../` changes. Those tables were measured on image `f8a15344e9fe3194...`, and the runs
below are different images with different declarations. Cite these by hash, not by label:

| run | image sha256 | differs from the published image in |
|---|---|---|
| `patched-stack-only/` | `de5c11808896dc1c...` | the declared stack ALONE (5 bytes, no executed byte) |
| `from-source-stack-and-arena/` | `9b9e7f383504fb31...` | stack + level-0 arena + a rebuilt SDK |

Both runs score three cases on all four arms with `--score-oracle-crashes`, so the fault SITE is
readable; CRASH-002 is carried as a positive control and must stay CAUGHT.

## CRASH-081: CAUGHT. The blocker was a 1 MiB stack declaration

    cause=25  reached=True  attribution=function
    ext/standard/array.c:3938 is ASAN's own file for this case, and its own line

CAUGHT on `block`, `object` and `exact` in BOTH runs; `stock` still cannot read it. The control
catches in both.

The case (bug #34277) is a use-after-realloc: `array_filter` holds `zval**` into the Zend argument
stack across a userland callback, `deep(4000)` inside the callback grows that stack so it reallocs
and MOVES, and the next loop iteration reads the dangling `input`. **The 4000-deep recursion is the
bug's own mechanism, not an accident of the trigger** -- which is why a stack ceiling blocked the
defect's precondition and the judge's `setup-fault` ("the fault came before the case reached its
defective access") was right for the right reason.

`zend_cap_block_realloc` never reallocs in place and calls `cap_retire_block(old)`, so the moved
block's payload aliases die. That predicted this catch before it was measured.

The attribution rests on `patched-stack-only/`, deliberately. That image raises the declared stack
from 1 MiB to 64 MiB by rewriting two quads of `.capstone_domreq` -- a section with no SHF_ALLOC
that the loader reads from the file -- and every PT_LOAD byte was verified identical to the
published image. One variable, same program. The from-source image cannot carry that claim:
`CAPSTONE_LEVEL0_ARENA_BYTES` is a compile-time constant, so raising the arena changes code
(77,720 bytes of PT_LOAD differ). Its job is to show the result survives a real build, which it
does.

Declaring a bigger stack is NOT free of gates. `application-image.c:74` enforces
`stack <= data - 256` and `data <= 256 MiB`; a first attempt raised the stack quad alone, and the
image inspector refused the image with ENOEXEC before anything ran, after which the cross-arm gates
refused to report verdicts. That is the gate chain working.

## CRASH-075: a platform ceiling, `CV_MAX_ARENAS`

Not caught, and now root-caused. Three variables were tried and the evidence separates them:

| change | effect |
|---|---|
| `--run-timeout` 40 -> 600 s | real progress: last region slot 4060 -> **8191** |
| level-0 arena 64 -> 512 MiB | **none**: `blocks=160803 live=160725 issues=160860` byte-identical |
| declared stack 1 -> 64 MiB | none (this case has no depth) |

The guest's last two lines:

    CAPSTONE_VM_ADD_OK slot=8191 base=0x3f7d820000 bytes=4096 ...
    ZEND_CAP_FAIL code=1 the heap would not lend a block a=0xb0 ...

8,192 distinct slots, max slot 8191 -- exactly `CV_MAX_ARENAS` (`runtime/virtual/wire.h:61`). A
176-byte `malloc` then fails. Byte-identical counters across an 8x arena change is the signature of
a hard table limit rather than memory exhaustion.

The shape matters more than the number: **8,089 of the 8,192 regions are 4 KiB**, 200.5 MiB in
total. mallocng takes one region per small group, so the table is spent on small objects long
before the bytes run short. `array_fill(0, 500000, 7)` -- bug #34066's way of forcing the same
argument-stack realloc -- needs far more groups than that.

Two routes, neither taken here because `CV_MAX_ARENAS` is an ABI constant in `wire.h` used by the
launcher's `maps[]` AND the kernel module's own array, which are the project lead's files:

1. raise `CV_MAX_ARENAS` (8192 -> 65536): simple, grows a fixed kernel array, buys 8x;
2. have mallocng request fewer, larger backing regions: fixes the class rather than moving the
   ceiling, and helps every port.

So CRASH-075 is a platform-scaling item, not a PHP-port defect and not a missed defect.

## What this implies for the counts, and what it does not

On a 64 MiB-stack image CRASH-081 reads CAUGHT, so the default temporal denominator would be 8 of
8 rather than 7 of 7, and the `--score-oracle-crashes` tally 27 of 29 rather than 26. **Those
numbers are not published here and must not be quoted from this directory**: they would require
re-running all 29 cases on one image, and the three cases measured here were run on two images,
neither of which is the published one. The published tables stand as measured.
