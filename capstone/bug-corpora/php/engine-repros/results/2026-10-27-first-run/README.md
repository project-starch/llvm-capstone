# First run from this corpus, 2026-10-27

The 68 cases of `php/engine-repros` on the virtual profile, dev `0b0b3414dca5` with the
patched port (`ports/php` at `af23230f70e5`; patch 0002 in, the sorting sanity control on),
one guest boot per arm per population on p13, judged by `tools/verdicts.py` through
`ports/php/app/run-realphp.py`. Pre-registered in the corpus README before the run; the
pre-registration commit is in every `inputs.json`. One bundle per arm here, the two
populations merged by `tools/merge-population-bundles.py` with both sources by hash.

## Readings

Denominators first. Of the 46 `asan-blind` cases, 36 can be read: 7 crash natively (the
judge reads them `out-of-denominator` unless an arm faults at the ASan site), 1 declares a
precondition no arm confirmed (CRASH-052, `ASAN_OPTIONS detect_stack_use_after_return`), 2
never returned (CRASH-075 and CRASH-092, the per-run watchdog; 075 is the mapping ceiling
the collaborator recorded, 092 blocks on a listener). Of the 22 `temporal-asan-visible` cases,
3 can be read: 18 crash natively and 1 faults before its defect (`setup-fault`).

| arm | configuration | `asan-blind`, of 36 | `temporal-asan-visible`, of 3 | catches by fault cause |
|---|---|--:|--:|---|
| control: stock (nothing revoked) | `php-real-stock` | 5 | 0 | 5 × cause 28 |
| `virtual-malloc` (mallocng, a lifetime per block) | `php-real-block` | 13 | 3 | 8 × 25, 5 × 28 (+1 unattributed) |
| `virtual-nested-pools` (a lifetime per `emalloc` object) | `php-real-object` | 26 | 3 | 21 × 25, 5 × 28 |
| `virtual-nested-exact` (+ bounds at the request) | `php-real-exact` | **30** | 3 | 21 × 25, 9 × 28 (+2 unattributed) |

Cause 25 is a load through a revoked capability -- a temporal catch, the lifetime mechanism
at work. Cause 28 is a bounds fault. The five cause-28 catches on the control arm are reads
past a bound the platform gives every arm, seam or not: past a whole mallocng block
(CRASH-094, 102, 120) or past a global's own bounds (CRASH-089, 090, `global-buffer-overflow`).
They are the platform's baseline, not the seam's. `virtual-nested-exact` adds four more
cause-28 catches (CRASH-016, 066, 091, 110) by bounding each payload to the request, which
is where PHP's `REAL_SIZE` rounding hid them.

No cause-24 catch anywhere: the untagged-pointer artefact that the earlier `qsort` swap
produced (the 2026-10-26 correction) is gone, and the sorting sanity control completed on
every arm.

## The two corrected cases, read for the first time

| case | stock | `virtual-malloc` | `virtual-nested-pools` | `virtual-nested-exact` |
|---|---|---|---|---|
| CRASH-022 (`usort` mutating the array it sorts) | MISSED | CAUGHT, cause 25 at `array.c:546` | CAUGHT, cause 25 at `array.c:546` | CAUGHT, cause 25 at `array.c:546` |
| CRASH-078 (`ksort`) | MISSED | CAUGHT, cause 25 at `array.c:171` | CAUGHT, cause 25 at `array.c:171` | CAUGHT, cause 25 at `array.c:171` |

Both are genuine temporal catches now -- a revoked capability loaded at ASan's own line --
and both are MISSED on the control, which is what a defect, as opposed to an artefact, looks
like. Their earlier "catches on every arm, stock included, cause 24" were the swap.

## Against the pre-registration

Held: the ordering is monotone on both populations (5 < 13 < 26 < 30; 0 < 3 = 3 = 3);
`virtual-malloc` catches exactly the readable `temporal-asan-visible` cases and, on
`asan-blind`, the use-after-frees whose block reaches a real `free()` (8 by revocation) plus
the platform's bounds catches; `virtual-nested-pools` catches the cache-absorbed
use-after-frees (21 by revocation); `virtual-nested-exact` adds the rounding-slack overflows;
CRASH-022 and CRASH-078 read as catches at the ASan site. The native crashers read
`out-of-denominator`.

Not as predicted, and reported as such: the prediction said `virtual-nested-pools` would
MISS the 12 rounding-slack overflows (CWE-125/787 in `asan-blind`) and `exact` would get
them. Measured: 5 of the 12 are caught on EVERY arm, the control included -- they cross a
bound the platform already enforces (three past the whole block, two past a global), so they
were never slack cases in the first place; `exact` catches 4 more; 2 fault on `exact` with
cause 28 but are `unattributed` (CRASH-008 at an address the symboliser could not place,
CRASH-073 whose ASan site is `zend_alloc.c` itself); 1 (CRASH-097, an overflowing write) is
MISSED on every arm. The four cells MISSED on `exact` are that one, a stack-address escape
(CRASH-051), an invalid free (CRASH-139) and one use-after-free (CRASH-084).

Three `unattributed` cells are recorded, not scored: CRASH-140 on `virtual-malloc` faults in
`Zend/zend_hash.c:998`, not ASan's file; CRASH-008 on `exact` faults at an address the
symboliser could not place; CRASH-073's ASan site is `zend_alloc.c` itself, which the judge
refuses to attribute by file (a fault in the allocator could be the seam's own).

## Against the collaborator's corrected 60-case result

Not the same denominator (46 + 22 here, their 31 + 29, on different ASan builds; the
corpus README says why), so no cell-for-cell comparison is claimed. The shape is the same:
their corrected 1 / 7 / 20 / 24 of 27 on `asan-blind` and 0 / 6 / 6 / 6 of 6 on
`temporal-asan-visible` against 5 / 13 / 26 / 30 of 36 and 0 / 3 / 3 / 3 of 3 here. The
stock catches differ in kind: theirs was the artefact, ours are bounds faults on the platform.

## What is here

`<arm>/verdicts.jsonl` (one judged Observation per case) and `<arm>/inputs.json` (the
configuration, the image by sha256, the platform by content -- compiler, QEMU, firmware,
kernel, module, launcher, runner -- the pre-registration commit, both populations' source
bundles by hash). The sensitivity scoring (`--score-oracle-crashes`) was not run.
