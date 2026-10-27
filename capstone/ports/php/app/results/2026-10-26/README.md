# Both populations, 2026-10-26

Answers the pre-registration at **`7437d8ffb35861aec45aa8a6e381a3bae7440ead`**, which every
bundle's `inputs.json` names in its `prereg` field. The platform these ran on is qualified:
`run.py` 42/42, recorded under `ports/php/app/results/platform-qualification-2026-10-26/`.

## Correction, 2026-10-27: CRASH-022 and CRASH-078 are a port artefact, not catches

The tables below are left as recorded. Eight of their CAUGHT cells are wrong, and the
explanation this file and the plan gave for two of them is wrong too.

**What was found.** Re-qualifying the port on dev `0b0b3414dca5`, a plain `sort(array(3,1,2))`
on a healthy program faulted with cause 24 (an untagged pointer used as a capability) in
`array_data_compare` at `ext/standard/array.c:324`, on the stock image and on every seam mode.
The cause is PHP's own `Zend/zend_qsort.c`: `_zend_qsort_swap` moves the `Bucket *` elements of
`zend_hash_sort`'s array as `int`s and then `char`s, which strips their tags; a 20-line probe
swapping two `malloc`'d pointers through `int` stores, then `char` stores, leaves both tags
clear. For a 20-element array `zend_qsort` swaps slots 0 and 9 *before* the first comparison,
so the comparator's first `f->pData` (or `f->nKeyLength`) load traps before any user code runs.

**Which cells.** CRASH-022's trigger is a `usort`; its ASAN line, `array.c:546`, is that first
load in `array_user_compare`. CRASH-078's is a `ksort`; its recorded site, `array.c:162`, is the
same load in `array_key_compare`. Every one of their cells, on all four arms and in all four
populations below, records cause 24 and never cause 25, and they are the only cause-24 catches
in `asan-blind` and (with CRASH-064, a different mechanism) in the temporal populations. The
fault lands before CRASH-022's comparator has done a single `unset()`, so "its freed block is
reused and overwritten with scalar data, which clears the tag" (this file, the plan at its
CRASH-022 passages, and the pre-registration's "faults on any build") describes an execution
that never happens. The platform the collaborator ran on differs from today's only in build
hashes; the pinned QEMU clears tags on scalar stores (`op_helper.c:3482` at `b290f565`), so
this held for those runs too.

**Corrected counts**, holding everything else fixed and reading the eight cells as
`NO-READING` (the instrument faulted before the defect):

| population | stock | block | object | exact | denominator |
|---|---|---|---|---|---|
| `asan-blind` CAUGHT | 1 | 7 | 20 | **24** | 27 (CRASH-022 leaves it) |
| `temporal-asan-visible` CAUGHT | 0 | 6 | 6 | 6 | 6 scoreable (CRASH-078 leaves it) |
| `temporal-scored-with-oracle-crashes` CAUGHT | 3 | 17 | 25 | 25 | as before, less CRASH-078 |

The arm ordering stays monotone. `exact` becomes 24 of 27, not 25 of 28; "the two `stock`
catches ... faulting on any build" is one catch (CRASH-102, cause 28, a bounds fault, which
stands) and one artefact.

**What changed so it cannot recur.** `app/patches/php-5.0.0-0002-zend-qsort-swap-keeps-the-capability.patch`
swaps pointer-sized units when the elements allow it (the PostgreSQL port's rule for
`sort_template.h`), so `sort`, `usort` and `ksort` work on every arm; and the runner's
`revocation-sanity` control now sorts three ways with a comparator that touches nothing, so an
arm whose sort faults reads "did not complete" on its control instead of catching. CRASH-022 and
CRASH-078 become measurable for the first time with that patch; their readings are owed by a
rerun of both populations on the prepared tree, and are not in this file.

The judge cannot tell this artefact from a detection on its own: attribution is by the ASAN
*file* (`run-realphp.py`, `verdicts.py`), and cause 24 is a legitimate detection class elsewhere
(CRASH-064), so the cause alone cannot be made a rule. The control is what carries the weight.

## `asan-blind` -- 31 cases: a cell-for-cell reproduction of the published result

| arm | CAUGHT | MISSED | unattributed | out-of-denominator | infra |
|---|---|---|---|---|---|
| `php-real-stock` | 2 | 26 | - | 2 | 1 |
| `php-real-block` | 8 | 19 | 1 | 2 | 1 |
| `php-real-object` | 21 | 7 | - | 2 | 1 |
| `php-real-exact` | **25** | 1 | 2 | 2 | 1 |

**25 of 28, and every one of the sixteen cells is identical to the published 2026-10-15 result**
(`docs/plans/2026-10-12-php-virtual-capstone-retarget.md:1104-1108`). That is after 879 upstream
commits, a firmware and monitor rebuilt at the recorded pin, a rewritten musl, and an allocator
seam whose block provider was moved onto a different ISA mechanism. The denominator is 31 less
CRASH-057 and CRASH-100 -- which the corpus's own pristine record says native PHP does not
complete -- and less CRASH-092, which produced no reading.

The arm ordering is monotone, 2 < 8 < 21 < 25, which is the cross-arm property the measurement
depends on. The two `stock` catches are CRASH-022 and CRASH-102, the two cases the
pre-registration named in advance as faulting on any build.

## `temporal-asan-visible` -- 29 cases, the complement

| arm | CAUGHT | MISSED | setup-fault | infra | out-of-denominator |
|---|---|---|---|---|---|
| `php-real-stock` | 1 | 6 | 2 | 1 | 19 |
| `php-real-block` | 7 | - | 2 | 1 | 19 |
| `php-real-object` | 7 | - | 2 | 1 | 19 |
| `php-real-exact` | 7 | - | 2 | 1 | 19 |

Seven of seven scoreable on the protected arms: CRASH-062, 070, 078, 083, 113, 117, 155.
`stock` catches only CRASH-078.

Two of the three refused cells were chased afterwards and are written up in
`targeted-stack-and-arena/`: **CRASH-081 is a catch blocked only by a 1 MiB declared stack**
(CAUGHT at `ext/standard/array.c:3938`, ASAN's own line, on a 64 MiB-stack image), and CRASH-075
is a platform ceiling -- `CV_MAX_ARENAS`, 8192 regions, 8,089 of them 4 KiB. Those runs are
DIFFERENT IMAGES; the tables here are unchanged and the improved counts are deliberately not
quoted into this file.

**There is no committed baseline for this population.** The run that preceded it was never
recorded -- no bundle, no table -- which the pre-registration says in its own words, so this is
the first recorded measurement of it. Nothing here should be described as having "changed".

## Sensitivity: the same records with the oracle-crashing cases scored

`temporal-scored-with-oracle-crashes/` re-judges the **same stored runs** with
`--score-oracle-crashes`, which lifts the `out-of-denominator` refusal so the 19 cases whose
pristine record is `SIGNAL:11` are read by their fault SITE. It is reported beside the default
scoring, never instead of it; the default table above is the published result.

| arm | CAUGHT | MISSED | unattributed | inconclusive | setup-fault | infra |
|---|---|---|---|---|---|---|
| `php-real-stock` | 4 | 6 | 7 | 8 | 3 | 1 |
| `php-real-block` | 18 | - | 8 | - | 2 | 1 |
| `php-real-object` | **26** | - | - | - | 2 | 1 |
| `php-real-exact` | **26** | - | - | - | 2 | 1 |

**What this does and does not say.** It is NOT 26 defects that native execution misses: native PHP
segfaults on all 19 of these, which is why they are out of the default denominator. What it
measures is *where* the failure lands. On 26 of 29 the capability fault lands in the case's own
ASAN-recorded file -- 24 of them by revoked-capability load (cause 25) -- whereas the native
signal lands wherever reuse happened to break. That is a localisation claim, and it is the one
these cases can support.

`object` and `exact` are identical, as they should be on a population that varies lifetime rather
than bounds; the ordering 4 < 18 < 26 = 26 is monotone.

Two things here are reported, not claimed. Three of `stock`'s four catches are cause 24, an
untagged pointer caught by the platform's baseline tag discipline with no revocation involved.
The fourth, CRASH-072, is cause 25 on an arm whose seam counter reads `revokes=0`, which can only
come from `heap.c`'s own revoke-on-free underneath the seam -- plausible, and unconfirmed, so it
should not be used before it is checked. CRASH-068 and CRASH-081 remain `setup-fault` under either
scoring and are the only two cases in this population that genuinely fault before their defect.

**This table is a correction.** Its first measurement reported CAUGHT=7 on every arm with all 19
cases reading `setup-fault`, and that was an instrument defect, not a result: `main()` ANDed the
oracle's premise into `site_licensed` while only `observe()`'s refusal was lifted, so `reached`
was false by construction for exactly the cases the flag exists to score, and the judge tests
`reached` before attribution. The faults had already been attributed to each case's own line. The
fix separates the two questions the code's own comment says must stay separate, is inert on the
default scoring (240 cells re-judged, 0 differ), and now carries an end-to-end must-fire control
in `test-run-realphp.py` that was negative-tested against the defect.

## Against the pre-registration

Held: no regression among the already-caught cases, which matters because the seam's block
provider was rewritten; `block` catching a strictly higher fraction; `stock` catching only
always-faulting cases; CRASH-100 staying out because `shrink-stack` is off; CRASH-008 and
CRASH-073 staying out of the denominator as `unattributed` rather than becoming catches.

Failed, and worth separating by kind:

- **Two were measurement surprises.** CRASH-046/075/081 did not become readings (that prediction
  has now failed twice, on two different stated causes), and CRASH-113 flipped from
  `unattributed` to CAUGHT. The raw faults for the first group are an out-of-bounds access
  BELOW an object (CRASH-081, `imm = -128`) and a revoked-capability access on a 2-byte object
  (CRASH-068), both landing inside the allocator -- a different mechanism from either cause
  predicted, and deliberately not given a third theory here.
- **Two were mine, not the instrument's.** The pre-registration predicted 24 caught and the 7
  OS-dependent cases staying unreachable. Both came from a Phase-A-era passage describing the
  PHYSICAL port's reach, while the published Phase B result -- 25 of 28, with those cases
  already reached -- sits twenty lines further down the same file. The predictions contradicted
  the committed baseline rather than reality, so they say nothing about this run. A
  pre-registration cannot protect against being written against the wrong prior.

Still open: whether any `exact` catch lies inside Option (iii)'s bounds granule, which requires
the fault addresses rather than the verdicts.

## Reading these bundles

`verdicts.jsonl` holds one judged `Observation` per line, so a reader needs neither this file nor
`verdicts.py` to see what was decided; `derive-verdicts.py --check` can re-judge them and will
refuse a stale verdict if the judge or `arms.json` changes. `inputs.json` records the platform by
content, and -- new in this run -- the emulator's `-cpu` string and the module's load arguments.
Those two decide whether the platform runs at all, and two runs differing only in them are not
the same measurement.
