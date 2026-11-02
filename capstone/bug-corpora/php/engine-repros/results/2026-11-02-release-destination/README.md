# Where a released block actually goes (2026-11-02)

Not an arm. This record gives no case a verdict and declares no configuration; it measures one
property of each defect -- when the defect's block is released, does it stop in Zend's per-size-class
cache, or does it reach libc `free()`? -- and it measures it inside the allocator rather than reading
it off a sanitizer's output.

## Why it was measured

This corpus's `population` field is *defined* by the `stockalloc` ASan run. That makes it unusable
as a test of the claim it is normally read as carrying -- that a block reaching `free()` is the one
an interposing detector can see -- because the axis and the test are the same measurement. The
obvious way out, classifying each case by reading its source for `efree` against a plain `free`,
does not work either: `_efree` means "the cache was consulted", not "the block was parked". The
cache holds 256 entries per size class and falls back to `free()` once full, so the destination
depends on how many blocks of that class are already cached at that moment. A sibling corpus records
the consequence directly: of its 56 cases, 40 are released through `_efree` and they split 12 that
reach `free()` against 28 that do not, and ten of the twelve have a payload of 80 bytes or less, so
a size threshold does not recover it either.

## How

Three builds, each doing only what it can do soundly. Their flags and hashes are in `inputs.json`.

**The probe.** `_efree` either puts the block in `AG(cache)` and returns, or falls through to
`ZEND_DO_FREE`, which is plain `free()` here (`ZEND_MM` is not defined). One unbuffered line per
release at each branch, plus `_erealloc`, the cache drain at request shutdown and the shutdown leak
sweep. Unbuffered because these cases crash, and a buffered tail is exactly the evidence that would
be lost.

**ASan as a locator, never as the classifier.** A build with the corpus's detector patches makes
every `efree` reach `free()`, so ASan reports the use-after-free, names the block and proves the
trigger reaches the use. 51 of 70 cases produce one. It decides nothing.

**One binary carrying both.** The decisive run has the stock cache ON, the probe in and ASan on. With
the cache on, ASan can only report a use-after-free on a block that genuinely reached libc, and the
probe says, at the same address in the same run, what happened to it. Nothing is inferred across
builds.

## Controls

A payload of 81 bytes or more is never cached -- 0 of 170 -- which is what
`cache_index = ((size+7)&~7)>>3 < 11` predicts from the source. A payload of 80 or less goes both
ways, 2729 cached against 790 passed through, which is the per-class cache filling. A run killed by
a signal still leaves its log. With `ZPROBE_OUT` unset the build writes nothing and is the stock
allocator. **One branch is unproven:** the shutdown leak sweep never fired in a control, and is
reported as untested; the three `nested-until-teardown` cases rest on the cache drain, which does
fire.

**The two instruments agree on every case they both see: 17 of 17, 0 disagreements.** The two they
do not are the answer for those two -- the probe has no record because the block never passed
through Zend's allocator -- and they are exactly the two cases `../plain-temporal-repros`
independently records as its only `nested=false` entries.

That cross-check earned its keep. Three cases first read as parked while ASan reported a
use-after-free on them; the probe was missing the cache drain at request shutdown. Instrumenting it
brought the disagreements to zero and separated `nested-until-teardown` as its own class.

## The result

| class | all 70 | the 55 temporal | what it means |
|---|--:|--:|---|
| `nested` | 32 | **31** | parked in `AG(cache)`; never reaches libc while the program runs. **Read from the pair of ASan builds, not from a `CACHED` line at the defect's block**: the patched build names the use-after-free, so the trigger reaches the use, and the decisive build with the cache on reports nothing at the libc boundary. For a parked block ASan has no address to name, so there is nothing for the probe's log to be joined against -- which is why the positive readings are the ones that reach libc |
| `nested-until-teardown` | 3 | **3** | parked during the run; returned to libc only when the cache is drained at request shutdown |
| `non-nested` | 14 | **14** | reaches libc `free()` during the run |
| `not-Zend` | 2 | **2** | never a sub-allocation: a `zend_strndup` constant name freed by a bare `free()`, and a persistent hash table released with `pefree()` |
| `no-release-observed` | 19 | 5 | no temporal release seen; 14 of the 19 are not temporal defects at all |

**34 of the 55 temporal cases are invisible to a mechanism that observes only system-allocator
events at the moment the defect is live.** 14 are not.

## What it says about this corpus's own axis

Over the 55 temporal cases:

Both rows are the axis **as each ASan run read it**, before this measurement was applied. The
`population_here` column of `matrix.tsv` is kept at that pre-correction reading on purpose: it is
what that instrument said, and a record that overwrote it could not be used to check the
instrument. The corpus's `population` field is the corrected one.

| axis as read by | `asan-blind` | `temporal-asan-visible` |
|---|---|---|
| the upstream corpus (gcc 13) | n=23: **22 `nested`, 1 unobserved, no exceptions** | n=28: 14 non-nested, 3 until-teardown, 2 not-Zend, **9 `nested`** |
| this corpus's run (gcc 15.2), as read | n=33: 22 `nested`, 2 unobserved, **9 in fact `non-nested`** | n=22: 5 non-nested, 3 until-teardown, 2 not-Zend, 9 `nested`, 3 unobserved |

(the upstream record puts a further 4 temporal cases in neither population; all 4 are unobserved
here.)

The upstream reading's `asan-blind` label is confirmed by direct measurement, with no exceptions.
**This corpus's own run read the property wrong on nine cases** -- CRASH-046, -062, -070, -075,
-081, -083, -113, -117 and -155. Eight are the ones the two instruments already disagreed about;
the ninth, CRASH-075, sits outside the 48 because `virtual-nested-exact` records it as `infra`
rather than a catch. The direct measurement settles every one of the nine in the upstream reading's
favour.

**APPLIED.** Those nine now carry the measurement's reading in the corpus: `population` moved from
`asan-blind` to `temporal-asan-visible`, 48/22 became 39/31, and each of the nine records what
placed it in `population_decided_by`. The rule is in `tools/build-cases.py`, which reads this
record's `matrix.tsv`; `--self-test` re-decides all 70 committed cases from their own recorded
readings and fails if any disagrees, which is the gate that holds on a host without the three
oracle runs. Those runs are not in the repository, so the nine case directories were rewritten by
the rule rather than by a full `--write`; **a full run on the oracle host must reproduce them byte
for byte**, and that is the one claim here a re-run still has to confirm.

With the nine corrected, this corpus's axis and the collaborator's agree on every case they both
place -- the 12 that still differ are all cases their run placed in neither population -- and the
baseline arm's split over the 48 reads 67 % against 33 % on both, where this corpus's own reading
previously gave 53 % against 52 %.

`temporal-asan-visible` is still over-inclusive in both readings: 9 of its cases are parked. The
label carries the meaning "the block reached `free()`" while, for those, its evidence supports only
"the as-shipped run produced some ASan output" -- of the 22 cases the stockalloc run placed there,
2 are `heap-use-after-free`, 2 an invalid free, and **18 a bare SEGV**. A SEGV is not evidence that
a block reached the system allocator, and the nine parked ones are nine of those 18. Correcting
that would mean taking cases OUT of the visible population, which this measurement cannot do on its
own: a parked block gives ASan no address to name, so there is no positive reading to put in place
of the label.

## Limits

One host, one toolchain. The mechanism being read -- a 256-entry per-class cache with a fall-through
-- is in the source and is not toolchain-specific, but the *occupancy* at a given release is a
property of the run, which is exactly why the two ASan builds disagreed. A second host would show
whether any case sits on that boundary. 19 cases produced no release to observe, and `matrix.tsv`
says so per case rather than assigning them a class.

Reproduced by `capstone/ports/php/app/release-probe/`.
