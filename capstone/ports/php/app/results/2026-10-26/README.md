# Both populations, 2026-10-26

Answers the pre-registration at **`7437d8ffb35861aec45aa8a6e381a3bae7440ead`**, which every
bundle's `inputs.json` names in its `prereg` field. The platform these ran on is qualified:
`run.py` 42/42, recorded under `ports/php/app/results/platform-qualification-2026-10-26/`.

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

**There is no committed baseline for this population.** The run that preceded it was never
recorded -- no bundle, no table -- which the pre-registration says in its own words, so this is
the first recorded measurement of it. Nothing here should be described as having "changed".

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
