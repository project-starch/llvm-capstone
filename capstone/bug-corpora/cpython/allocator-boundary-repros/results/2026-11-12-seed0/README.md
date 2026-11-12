# cpython/allocator-boundary-repros on the virtual profile, hash seed 0, 2026-11-12

Both virtual arms, measured again with one change to the invocation of `../2026-11-11-t1800`:
every guest process runs with `PYTHONHASHSEED=0` (`run-virtual.py --guest-env PYTHONHASHSEED=0`).
The images, the platform, the session form and `--timeout 1800` are those of 2026-11-11.
`corpus.json` names the two bundles of this record as the declared bundles.

| arm | image | workload and controls (uaf-block, bounds-block) | cases | against `../2026-11-11-t1800` | against `../2026-10-20-review-t1800` |
|---|---|---|---|---|---|
| `virtual-nested-pools` (`virtual-cpython-pools`) | `be62bdc9c306`, the image of 2026-11-11 | ok; fault, fault, as declared | CAUGHT 28, MISSED 3, NO-READING 1 | 30 of 32 rows equal in verdict, cause, function and offset; 02 and 03 keep the verdict at another site | 28 of 32 rows equal in verdict, cause and function |
| `virtual-malloc` (`virtual-cpython`) | `6304c04f9f8e`, the image of 2026-11-11 | ok; complete, complete, as declared | CAUGHT 17, MISSED 14, NO-READING 1 | 30 of 32 rows equal in verdict, cause, function and offset; 04 and 09 differ | 29 of 32 rows equal in verdict, cause and function |

## The rule

The declared bundles of this corpus are measured with `--guest-env PYTHONHASHSEED=0`. The runner
passes the variable to every guest process: the workload, both controls, each trigger, each negative
control and each probe. Nothing else about the run changes. Seed 0 was fixed before the first seeded
run of this record, and no other seed was run.

The unseeded reading of rows 02, 03 and 04 was an artefact of one process's hash key. The three
triggers hand a `bytes` field name to a `%U` format, and `_copy_characters` then reads the `bytes`
object through the `str` layout. On this ABI `PyVarObject` is 48 bytes, so `PyBytesObject.ob_shash`
lies at offset 48, which is the offset of `PyASCIIObject.state` (measured with the build's compiler
and headers: `ob_size` 32, `ob_shash` 48, `ob_sval` 56; `length` 32, `hash` 40, `state` 48). The word
that tells the copy how wide a character is, and where the characters lie, is therefore the cached
hash of the field name. CPython keys that hash per process unless `PYTHONHASHSEED` fixes the key.
The copy function loads the word with `lw 0x30(a2)` and tests bit 5 (compact), bit 6 (ASCII) and
bits 2 to 4 (kind). One process copies one-byte units from inside the object, the next copies
two-byte units past its end, and the next takes its data pointer from the payload. On x86-64 the
same field lies inside the payload (`ob_sval` at 32, `state` at 32), so the host ASan run that
recorded the fault sites always saw `0xffffffff` there and has no such spread.

The old reading: unseeded, rows 02 and 04 took both verdicts from run to run on both platforms.
`../2026-11-11-t1800/README.md` has the spread table (ten runs of `virtual-nested-pools` over the
converted and the unconverted tree: 02 CAUGHT in six, 04 CAUGHT in four; cause 28 in
`_copy_characters` or in `memcpy`). `../2026-10-20-review-t1800` read 02 MISSED and 04 CAUGHT on
`virtual-nested-pools`, and 02 and 04 MISSED on `virtual-malloc`. Those records stay as they are.

## The repeats: five runs per arm, cases 02, 03 and 04

`repeats/<arm>/run-N` holds ten runs of `--only 02,03,04 --guest-env PYTHONHASHSEED=0`, each one
boot with the workload and both controls first. They ran before the full arms, because they decide
whether the rule does what it is for.

| arm | 02 | 03 | 04 |
|---|---|---|---|
| `virtual-nested-pools` | CAUGHT, cause 28 in `memcpy+0x1c0`, by control: 5 of 5 | CAUGHT, cause 28 in `memcpy+0x1c0`, by control: 5 of 5 | MISSED, the trigger ended with exit 1: 5 of 5 |
| `virtual-malloc` | MISSED, the trigger ended with exit 1: 5 of 5 | MISSED, the trigger ended with exit 1: 5 of 5 | MISSED, the trigger ended with exit 1: 5 of 5 |

The five runs of an arm agree in every field of the observation except addresses. The controls
read as declared in all ten (fault, fault on the pools arm; complete, complete on the stock arm).

The guest saw the seed in all ten runs and in both full arms: `hash_randomization` 0,
`pythonhashseed` "0", `hash_abc` 1453079729188098211 every time. The kernel command line is
`root=/dev/vda ro`, without `norandmaps`, so the address space layout still differs from process to
process: the 20 faulting processes of the pools repeats ran at 18 different code bases
(`0x3f7f000000` to `0x3fbc000000`). The three rows do not depend on it.

## What seed 0 selects in rows 02, 03 and 04

The build hashes with FNV (`HAVE_ALIGNED_REQUIRED` in its `pyconfig.h`; the guest's `hash_abc` is
FNV of "abc" under a zero key). Under a zero key the hash of the field name, and with it the misread
`state` word, is a constant of the trigger:

| trigger | field name | low byte of the hash | misread as | the copy |
|---|---|---|---|---|
| 02, 03 | `b'\xff' * 64`, a 121-byte object | `0x65` | kind 1, compact, ASCII | `memcpy` of 64 bytes from offset 64, so 7 bytes past the object |
| 04 | `b'\xff' * 524288` | `0x9d` | kind 7, not compact | data pointer from 16 payload bytes at offset 96; no character is read |

**02 and 03.** On `virtual-nested-pools` the block's capability ends at the request, and the copy
faults on the first byte past it: `Cap mem access OOB`, a one-byte load at `memcpy+0x1c0`, address
equal to the upper bound, bounds of length `0x79` (121), return address in `_copy_characters`. The
ten faults of the repeats (two per run) have this shape, at the same offset in the page and at a
different base in every run. `memcpy` is not one of the declared fault sites (`_copy_characters`),
so the runner attributes by the negative control, which ran clean each time. On `virtual-malloc`
the same copy completes: the message holds 56 bytes of `0xff`, the terminator and 7 bytes of the
block's own tail (121 bytes in a 128-byte size class). In case 03 those 7 bytes differ from run to
run, and the reading does not depend on them.

**04.** The misread data pointer is 16 bytes of `0xff` without a tag, so its address is 2^64 - 1.
The copy loop computes its end from that address, the end wraps below the start, and the loop body
never runs (`bgeu` at `_copy_characters+0x658` and `+0x68c` in the nested image). The trigger then
prints a field name of 524288 NUL characters, the fresh destination buffer, and ends with exit 1.
All ten repeats and both full arms show this. **Under seed 0 the run of case 04 performs no
out-of-bounds access on either arm.** The judge reads the row MISSED, as it reads rows 06 and 17,
whose binaries contain no out-of-bounds access either. The row says nothing about the bounds of the
524321-byte object. The unseeded CAUGHT readings of 04 came from processes whose hash selected a
two-byte or four-byte copy past the mapping.

## The full arms

One boot per arm, 56 steps each: the workload, both controls, 32 triggers and every negative
control and probe. The command is the one of 2026-11-11 plus `--guest-env PYTHONHASHSEED=0`. Both
boots completed 56 of 56 steps, and the controls read as declared. The images are the images of
2026-11-11, so an equal row is equal down to the offset of the fault in its function.

### Against `../2026-11-11-t1800`

| arm | case | 2026-11-11, unseeded | this record | why |
|---|---|---|---|---|
| `virtual-nested-pools` | 02 | CAUGHT, cause 28 in `_copy_characters+0x664`, by function | CAUGHT, cause 28 in `memcpy+0x1c0`, by control | the seed: the one-byte copy, where that process's hash had selected a two-byte copy |
| `virtual-nested-pools` | 03 | CAUGHT, cause 28 in `_copy_characters+0x5c0`, by function | CAUGHT, cause 28 in `memcpy+0x1c0`, by control | the seed, as 02 |
| `virtual-malloc` | 04 | CAUGHT, cause 28 in `_copy_characters+0x65c` | MISSED, the trigger ended with exit 1 | the seed: no character is read |
| `virtual-malloc` | 09 | NO-READING:inconclusive | CAUGHT, cause 28 in `unicodekeys_lookup_unicode+0x184`, by control | not the seed, see "Row 09" |

The other 30 rows of each arm are equal in verdict, reason, cause, function and offset. Row 04 on
`virtual-nested-pools` reads MISSED in both bundles; in the unseeded bundle that was one of the two
outcomes its samples show. **Row 09 on `virtual-malloc` is the one row other than 02, 03 and 04
that reads differently, and the seed does not explain it.**

### Against `../2026-10-20-review-t1800`

| arm | case | 2026-10-20 | this record | why |
|---|---|---|---|---|
| `virtual-nested-pools` | 02 | MISSED, the trigger ended with exit 1 | CAUGHT, cause 28 in `memcpy+0x1c0`, by control | the seed |
| `virtual-nested-pools` | 03 | CAUGHT, cause 28 in `_copy_characters` | CAUGHT, cause 28 in `memcpy+0x1c0`, by control | the seed: same verdict, the copy the seed selects |
| `virtual-nested-pools` | 04 | CAUGHT, cause 28 in `_copy_characters` | MISSED, the trigger ended with exit 1 | the seed: no character is read |
| `virtual-nested-pools` | 18 | MISSED, the trigger ended with exit 1 | CAUGHT, cause 28 in `mmap_ass_subscript+0x27c` | the runtime's `mremap` of 2026-10-22 |
| `virtual-malloc` | 09 | NO-READING:inconclusive | CAUGHT, cause 28 in `unicodekeys_lookup_unicode+0x184`, by control | not the seed, see "Row 09" |
| `virtual-malloc` | 13 | NO-READING:infra, capacity: `MemoryError` | MISSED, `MemoryError` declared as the defect's own uncaught read | the case's declaration of 2026-10-22 |
| `virtual-malloc` | 18 | MISSED, the trigger ended with exit 1 | CAUGHT, cause 28 in `mmap_ass_subscript+0x27c` | the runtime's `mremap` of 2026-10-22 |

Rows 02 and 04 on `virtual-malloc` read MISSED in both. The other rows are equal in verdict, reason,
cause and function. Their offsets differ in 23 rows of the pools arm and 12 of the stock arm,
because the two records have different images.

**13 and 18** read as in `../2026-11-11-t1800`, whose README gives the evidence: row 13 makes the
same observation as on 2026-10-20, a `MemoryError`, and its `case.json` declares
`memory_error_is_the_defect` since 2026-10-22. Row 18 needs `mmap.resize`, the runtime serves
`mremap` since 2026-10-22, and the write past the shrunk mapping faults on the mapping's bounds in
a declared fault site. Neither depends on the seed, and both rows are equal to 2026-11-11 down to
the offset.

## Row 09 on `virtual-malloc`: a reading the seed does not fix

Case 09 stores NULL past a frame (`frame->localsplus[i] = NULL` in a loop bounded by
`co_nlocalsplus`, in a frame sized by `co_framesize`). The stock arm does not bound a frame, so the
stores succeed, and the row reads what the interpreter does afterwards.

| run | seed | 09 on `virtual-malloc` |
|---|---|---|
| `../2026-10-20-review-t1800` | none | NO-READING:inconclusive |
| `../2026-11-11-t1800` | none | NO-READING:inconclusive: `Fatal Python error: _PyEval_EvalFrameDefault: Executing a cache`, abort |
| `virtual-malloc` of this record | 0 | CAUGHT, cause 28 in `unicodekeys_lookup_unicode+0x184`, by control |
| `row-09/virtual-malloc/only-09-1` to `-5` (`--only 09`) | 0 | NO-READING:inconclusive, the same abort: 5 of 5 |
| `row-09/virtual-malloc/cases-01-to-09` (`--only 01,...,09`) | 0 | NO-READING:inconclusive, the same abort |

The fault of the full arm is an 8-byte load in a dictionary lookup, `0x1f8` bytes past the upper
bound of a region of `0x100000` bytes (a pymalloc arena as mallocng bounds it). It is not the
defective store and not in the case's fault sites (`_PyFrame_Initialize`,
`_PyEvalFramePushAndInit`). The runner counts it because the negative control ran clean. By the
corpus README's criterion it is a consequence of the store, not a rejection of it.

The reading is 1 of 7 seeded runs of this image. The six others ran after the full arm to test two
explanations. Five boots with case 09 alone exclude the seed. One boot with cases 01 to 09 gives
case 09 the same processes before it as the full arm, and it reads the abort too, so the history of
the boot does not select the fault either. Rows 01 to 08 of that boot equal the full arm. What
selects the fault is not established. The address space layout is not pinned (no `norandmaps`), and
the trigger runs an asyncio event loop. `../2026-10-19-vp2-trimmed-pinned` already lists 09 among
the cells of this arm that vary under seed and `norandmaps` together, and found the residual
consistent with host timing.

The declared bundle keeps the reading its run produced. Replacing the run by another one would
choose a run by its outcome. A reader of the `virtual-malloc` tally should count row 09 as a
sample: CAUGHT 17 with it, 16 without.

## Timing

The pools arm took 1 h 56 min and the stock arm 1 h 30 min, side by side and beside a third boot
for part of the time. The longest step is case 21 in both, about 760 s and 840 s (sampled every
10 s from the console), inside the 1,800 s budget. A session does not enforce the budget per step.

## Platform

The kit, the CPU line and the module are those of `../2026-11-11-t1800` (qemu `2d5a7107890e`,
`CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`, module loaded
without parameters). `inputs.json` carries every file hash and the guest's reading of its
environment. The judge hash differs from the 2026-11-11 bundles (`5d805ef4d83d` against
`9011d1cf7ef3`), because `tools/verdicts.py` changed on 2026-11-12 for another corpus.
`derive-verdicts.py --check` re-judges every stored observation of this corpus with the current
judge and finds no difference.

Each run went through `tools/board/appvm-session.py` with the kit as a virtual kit (collect, boot,
read) and `--timeout 1800`, on a lock file of its own, at most three boots at a time.

## Images and library

The two images are the files that ran on 2026-11-11, byte for byte (`be62bdc9c306` nested,
`6304c04f9f8e` stock; the runner records the hash of the file it ran in every observation). The
build directories of 2026-11-11 no longer exist, so the library (`Lib`, compiled by the native
3.13.7) and the SDK records come from a rebuild of the same tree with the same command at another
path:

    CAPSTONE_SDK_CORPUS=1 CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin CPY_TEST_CAPI=1 [CPY_SUBLET=1] \
        capstone/ports/common/application/build-virtual.sh cpython OUT

The rebuilt images were not run (`4b5799ff2bb3` nested, `be416f3a136b` stock). They differ from the
recorded ones in the 60 source paths that name the build directory, in the build time (`__TIME__`)
and in the data addresses that shift with the shorter paths: `.text` has the same size, every
function has the same address, and `.rodata` is 592 bytes shorter. An image of this port is therefore not reproducible
byte for byte from the recipe, and a later record that needs the same image must keep the file.
The staged library differs from the 2026-11-11 stage in one file, `_sysconfigdata__linux_.py`,
which carries the build directory's path.
