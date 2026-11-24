# cpython/allocator-boundary-repros on the virtual profile, 2026-11-11

Both virtual arms, measured again after patch 0014 and the processor model changed their lifetime
instructions. obmalloc now derives every block from the arena as mallocng returned it
(`arena_life` is gone) and ends a block with `CREVOKE(block)`.

| arm | build | workload and controls (uaf-block, bounds-block) | cases | against `../2026-10-20-review-t1800` |
|---|---|---|---|---|
| `virtual-nested-pools` (`virtual-cpython-pools`) | `CPY_TEST_CAPI=1 CPY_SUBLET=1 build-virtual.sh cpython`, corpus SDK | ok; fault, fault, as declared | CAUGHT 28, MISSED 3, NO-READING 1 | 29 of 32 rows equal in verdict, cause and site |
| `virtual-malloc` (`virtual-cpython`) | `CPY_TEST_CAPI=1 build-virtual.sh cpython`, corpus SDK | ok; complete, complete, as declared | CAUGHT 17, MISSED 13, NO-READING 2 | 29 of 32 rows equal in verdict, cause and site |

`corpus.json` still names the 2026-10-20 pair as the declared bundles. The corpus README's
case-by-case review is written against that pair, and six cells read differently here for the
reasons this page gives. None of them comes from the conversion.

Successor: since 2026-11-12 `corpus.json` names `../2026-11-12-seed0`, the same images measured with
`PYTHONHASHSEED=0`. Its README explains why rows 02 and 04 spread here and how they read under the seed.

The 29 equal rows of `virtual-nested-pools`: cause 28 at the same declared function for the spatial rows 01, 03, 05, 07 to
10 and 12 to 16, cause 25 at the same function for the temporal rows 19 to 32 (19 by its control
pairing), MISSED for 06 and 17, NO-READING:not-reached for 11 (`ModuleNotFoundError`, no `_ctypes`).

The 29 equal rows of `virtual-malloc` include every fault with its cause and site, among them the
three rows that mallocng alone serves (19 in `fnv`, 20 in `memcpy`, 21 in `little2_toUtf8`, all
cause 25).

## The rows that differ, `virtual-nested-pools`

| case | 2026-10-20 | 2026-11-11 | why |
|---|---|---|---|
| 02 | MISSED, the trigger ended with exit 1 | CAUGHT, cause 28 in `_copy_characters` | neighbouring memory, see below |
| 04 | CAUGHT, cause 28 in `_copy_characters` | MISSED, the trigger ended with exit 1 | neighbouring memory, see below |
| 18 | MISSED, the trigger ended with exit 1 | CAUGHT, cause 28 in `mmap_ass_subscript` | the runtime's `mremap` of 2026-10-22 |

**02 and 04** are the two rows the corpus README records as depending on the contents of
neighbouring memory: `_copy_characters` copies a length taken from the field the overread fetched,
and CPython randomises what lies there per process. Both bundles ran with hash randomisation on
(`inputs.json`: `hash_randomization: 1`, no `PYTHONHASHSEED`). Each row takes both outcomes on both
platforms. The samples are runs of `--only 02,04` (the workload and both controls first, as
declared, in each), from the image of this bundle on this platform and from the unconverted tree's
image on that tree's platform (`dev-control/` and `samples/dev-N`: patch 0014 with the two-operand
`CREVOKE`, qemu `35529815bf4b`, CPU line with `x-capstone-supervisor-slots=32` for its module):

| run | 02 | 04 |
|---|---|---|
| this bundle | CAUGHT, 28 in `_copy_characters` | MISSED |
| `samples/branch-1` | MISSED | MISSED |
| `samples/branch-2` | MISSED | MISSED |
| `samples/branch-3` | CAUGHT, 28 in `_copy_characters` | CAUGHT, 28 in `_copy_characters` |
| `samples/branch-4` | CAUGHT, 28 in `_copy_characters` | CAUGHT, 28 in `memcpy` |
| `samples/branch-5` | CAUGHT, 28 in `_copy_characters` | MISSED |
| `dev-control` | CAUGHT, 28 in `_copy_characters` | CAUGHT, 28 in `_copy_characters` |
| `samples/dev-1` | CAUGHT, 28 in `memcpy` | MISSED |
| `samples/dev-2` | MISSED | CAUGHT, 28 in `_copy_characters` |
| `samples/dev-3` | MISSED | MISSED |

**18** is `mmap.mmap.__setitem__` after a resize from inside `__index__`. Its trigger needs
`mmap.resize`, and the runtime serves the public `mremap` since 2026-10-22
(`runtime/virtual/mapping.c`, whose comment names this case). The 2026-10-20 image predates
that (runtime revision `6a6ae69e0747`), and its trigger ended with exit 1. Now the resize
succeeds, the write lands past the shrunk mapping, and the mapping's bounds reject it:
`Cap mem access OOB`, bounds `[..227000, ..228000)`, address `..228fff`, cause 28 in
`mmap_ass_subscript`, a declared fault site. `mapping.c` is byte-identical before and after the
conversion, and pymalloc plays no part in this row. `dev-control/virtual-nested-pools` (`--only
02,04,18` on the unconverted tree and its platform) reads this row the same way: CAUGHT, cause 28
in `mmap_ass_subscript`.

## The rows that differ, `virtual-malloc`

| case | 2026-10-20 | 2026-11-11 | why |
|---|---|---|---|
| 04 | MISSED, the trigger ended with exit 1 | CAUGHT, cause 28 in `_copy_characters` | neighbouring memory, as on the other arm |
| 13 | NO-READING:infra, capacity: `MemoryError` | MISSED, `MemoryError` declared as the defect's own uncaught read | the case's declaration of 2026-10-22 |
| 18 | MISSED, the trigger ended with exit 1 | CAUGHT, cause 28 in `mmap_ass_subscript` | the runtime's `mremap` of 2026-10-22, as on the other arm |

**13** made the same observation in both bundles, a `MemoryError`. Its `case.json` declares
`memory_error_is_the_defect` since 2026-10-22, and `../2026-10-22-batch-declared` already reads the
row MISSED for that reason.

## Timing

One boot per arm, 56 steps each (the workload, two controls, 32 triggers, and every negative
control and probe, which a session collects unconditionally). The pools arm took 2 h 51 min,
the stock arm 1 h 42 min beside two other boots. The longest step is case 21 in
both, about 1,040 s and 770 s, inside the 1,800 s budget of the earlier bundle. A session does not
enforce the budget per step; the step times were sampled from the console during the boots.

## Lifetime instructions in the image

| image | `CDERIVE` | `CREVOKE`, one operand | `CREVOKE`, two operands |
|---|---:|---:|---:|
| nested, patch 0014 of this date | 12 | 5 (`_PyObject_Free`, `_PyObject_Realloc`, mallocng's `free` and `realloc`) | 0 |
| nested, patch 0014 before the conversion (same compiler) | 14 | 0 | 5 |
| stock | 6 | 3 (mallocng) | 0 |

## Platform

The kit is the tree's virtual platform on the model with the one-operand `CREVOKE` and `CRENEW`.
`inputs.json` carries every file hash and the guest's own reading of its environment. Two settings
are not in it:

* CPU properties: `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`,
  so the `-cpu` line reads
  `rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,x-capstone-test-csrs=true,x-capstone-exact-bounds=true`.
  The earlier bundle records `exact_bounds: true`.
* Module: loaded without parameters (`unpinned=0`, `supervisor_slots=1`).

Each arm went through `tools/board/appvm-session.py` with the kit as a virtual kit (collect, boot,
read) and `--timeout 1800`. The earlier bundle ran on a persistent VM with a 9p share; here the
library and the triggers lie on the boot's read-only stage disk.

## Build

    CAPSTONE_SDK_CORPUS=1 CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin CPY_TEST_CAPI=1 CPY_SUBLET=1 \
        capstone/ports/common/application/build-virtual.sh cpython OUT

`CAPSTONE_SDK_EXTRA_CODEGEN` records the corpus codegen in the relink SDK's `sdk.json`, which the
runner's builtins gate reads; the nested image is byte-identical with and without it
(`be62bdc9c306`). The stock arm is the same command without `CPY_SUBLET=1` (image `6304c04f9f8e`).
Patch 0014 is `ff1438e007c0`.
