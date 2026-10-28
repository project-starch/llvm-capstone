# php/engine-repros

PHP 5.0.0 memory-safety defects -- in the Zend engine and in `ext/standard` -- as the scripts
that trigger them, run through the real interpreter on the virtual profile
(`ports/php/app`). What this corpus asks of each case is **which layer makes the defect
visible**: the Zend allocator hides it (its size-class cache parks freed blocks so `free()`
is never reached, or its 8-byte rounding leaves the overflow in slack), or `efree` reaches
`free()` and the libc boundary sees it. That is decided per case by measurement, never by
reading the defect: ASan with PHP's allocator patched against ASan with it as shipped.

| population | cases | what it is | who can catch it |
|---|--:|---|---|
| `asan-blind` | 46 | ASan reports it only with the allocator patched; as shipped the run reads OK. By CWE: 30 use-after-free, 10 out-of-bounds reads and 2 writes, 2 stack-address escapes, 1 free of non-heap memory, 1 uninitialised read; 7 of the 46 crash natively | selected by measurement, not by mechanism: what each arm sees is in the arms' records. The first run has the seam arms (`virtual-nested-pools`, `virtual-nested-exact`) at 26 and 30, the outer allocator alone (`virtual-malloc`) at 13 -- 8 use-after-frees whose block does reach `free()` plus 5 bounds catches the platform gives every arm |
| `temporal-asan-visible` | 19 | CWE-416 that ASan reports with the allocator as shipped (the `stockalloc` run) AND that the patched build classifies as a use-after-free: the block reached `free()`; 17 of the 19 crash natively, so the judge's default denominator holds two of them and the `--score-oracle-crashes` sensitivity scoring reads the rest | the outer allocator by itself, `virtual-malloc` (musl mallocng, a lifetime per block), for a block that reaches `free()` and is not parked in `AG(cache)`; the seam arms for the parked ones. Which is which is in the arms' records |

The two do not overlap. Every case is a client defect in PHP's own code; none is a bug in
libc or in `zend_alloc.c`, which is witness, not culprit.

## Where the cases come from, and what is ours

`tools/build-cases.py` writes every directory here. Its inputs are the collaborator's
`php-5.0.0-bug-corpus` (185 cases with per-case metadata: CWE, the defective `file:line`,
the php-src fix commit, how collected; zip sha256 in every `PROVENANCE.md`; not in this
repository) and **our own three native runs** of its 127 default-build cases with the
corpus's own `scripts/run-corpus.py`, made on p12 (gcc 15.2, libasan 15) from the museum
tarball by the corpus's recipes: `asan` (its two `zend_alloc.c` detector patches), `stockalloc`
(the same without those patches -- the allocator as shipped under ASan; the corpus does not
ship this run, and the populations are *defined* by it) and the plain build.

The result is 65 cases (46 + 19) where the collaborator's instrument measured 60 (31 + 29):
**16 cases sit in a different population on our ASan than on theirs**, CRASH-064 (a
handle-after-free that both ASan builds report only as SEGV, so no build names it a
use-after-free) is in neither population, two of theirs end as
shipped in PHP's own fatal error with no report and so are in neither population here -- on
gcc 13's ASan the freed block
of CRASH-081 (for one) reached `free()` and was reported; on gcc 15's it reads OK with the
allocator as shipped -- and CRASH-100 (needs an HTTP listener) is absent. "ASan-blind" is a
property of the detector build as much as of the defect, which is one more reason the
population is recomputed from recorded runs rather than carried as a list. Each `PROVENANCE.md`
says where our run and the corpus's record disagree, case by case.

## A case directory

```
NN_<fix12>_crash-nnn/
  trigger.php      the corpus script, byte for byte
  oracle.json      what run-realphp.py reads: population, ASan's first PHP frame (the
                   expected site), the pristine run's outcome, rc and stdout, the crash flag,
                   and the sha256 of the three result files it was derived from
  case.json        the framework's script-trigger record; derive-verdicts.py fills the arms
  PROVENANCE.md    the corpus record verbatim, our oracle, every disagreement
```

The directory prefix is the php-src fix commit (12 hex digits); the suffix the corpus id.

## Running it

```sh
python3 capstone/ports/php/app/run-realphp.py --corpus capstone/bug-corpora/php/engine-repros \
  --population asan-blind --image php-real-seam.dom --probe probe-seam.dom --llvm-bin "$CAPSTONE_LLVM_BIN" \
  --qemu ... --images ... --adapter ... --out results/<stamp>/asan-blind
```

One guest boot per arm, every case its own `capstone-vexec` process. The runner drives four
configurations of one seam image (`tools/arms.json` `php-real-stock|block|object|exact`,
`PHP_CAP_MODE` 0/4/6/7) and writes one bundle per arm under the framework's names
(`virtual-malloc` = block, `virtual-nested-pools` = object, `virtual-nested-exact` = exact);
the stock run is the judge's control for `reached` and keeps its own name. A run whose
transcript shows a defaulted arm (`ZEND_CAP_MODE_DEFAULTED`) or another arm's mode is refused.
`corpus.json`'s `verdict_bundles` names the current bundles, `derive-verdicts.py` writes the
verdicts into each `case.json`, and `check-corpus.py` re-judges them.

## Pre-registration, before the first run from this corpus (2026-10-27)

Written before any arm ran over these 68 cases on the patched port (`ports/php` at this
commit, dev `0b0b3414dca5`), from the collaborator's corrected 60-case readings and the
mechanism, so that a surprise is a surprise:

- `virtual-malloc` (per-block mallocng): CAUGHT on the `temporal-asan-visible` cases the judge
  can read (the block reached `free()`, and mallocng retires it), MISSED on the `asan-blind`
  use-after-frees the cache absorbs except those that eventually reach a real free (the
  collaborator read 7 of 27); nothing on rounding-slack overflows.
- `virtual-nested-pools` (a lifetime per `emalloc` object): CAUGHT on the `asan-blind`
  use-after-frees (the collaborator read 20 of 27 on the smaller set); MISSED on the 12
  rounding-slack overflows, which need bounds at the request.
- `virtual-nested-exact`: `virtual-nested-pools` plus the slack overflows; the highest count.
- CRASH-022 and CRASH-078, whose earlier "catches" were PHP's own `qsort` swap, read for the
  first time; a CAUGHT there is at the ASan site or it is not a catch.
- The 7 + 17 cases that crash natively (7 of the 46 asan-blind, 17 of the 19 temporal-asan-visible,
  counted from the oracle files; 23 of them CWE-416, the 24th is CRASH-153, CWE-457) read `out-of-denominator` on every arm under the
  judge's default scoring, whatever the arm did; the sensitivity scoring
  (`--score-oracle-crashes`) reads them by fault site and is reported beside, never instead.
- The 16 cases placed differently from the collaborator's populations carry no prediction
  beyond the mechanism above; whatever they read is a reading of this instrument.

## Results

Three records, each a run of the whole corpus on the virtual profile, one bundle per arm with the two
populations merged (`tools/merge-population-bundles.py`), every table computed from the bundles by
`tools/result-tables.py` and written into that record's README, never typed:

- `results/2026-10-27-run7/`: lane B's measurement (the corpus at `672a0deffaa2`; runs 5, 6 and 7 agree cell for cell).
- `results/2026-10-28-run10/`: lane C's measurement (the seam on the libc's reclaim, `8569fb7194dc`): against run7 no CAUGHT changed.
- `results/2026-10-28-run14/` and `results/2026-10-28-run15/`: the follow-up lane's measurement (`1ed7d8c601e2`, then `009ca7b34b00` with the launcher that names its image when a run ends): the 8 MiB-stack SDK with line tables for the musl objects, the launcher that records the frame chain, the putenv probe. run15 is cell-identical to run14. **run15 is the current record:** `corpus.json`'s `verdict_bundles` names its arm bundles, `derive-verdicts.py` wrote their verdicts into every `case.json`, and `check-corpus.py` re-judges them. run14 is cell-identical to run13, which it replaces only because run13's two boots recorded different probe lists and could not be merged.

Over the 49 CWE-416 cases, from `results/2026-10-28-run15/README.md` (identical to run14's):

**Default scoring** (the 23 native crashers out of the denominator):

| arm | CAUGHT | MISSED | NO-READING |
|---|--:|--:|---|
| stock | 1 | 21 | infra 1, not-reached 2, out-of-denominator 24 |
| system allocator | 10 | 11 | infra 1, not-reached 2, out-of-denominator 24, unattributed 1 |
| nested | 23 | 0 | infra 2, out-of-denominator 24 |
| nested exact | 23 | 0 | infra 2, out-of-denominator 24 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 1 of 19 / 0 of 30
- system allocator: 2 of 19 / 8 of 30
- nested: 2 of 19 / 21 of 30
- nested exact: 2 of 19 / 21 of 30

**Sensitivity scoring** (every case scored by fault site; reported beside the default, never instead):

| arm | CAUGHT | MISSED | NO-READING |
|---|--:|--:|---|
| stock | 2 | 21 | infra 2, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 3 |
| system allocator | 23 | 11 | infra 1, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 2 |
| nested | 46 | 0 | infra 2, out-of-denominator 1 |
| nested exact | 46 | 0 | infra 2, out-of-denominator 1 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 2 of 19 / 0 of 30
- system allocator: 10 of 19 / 13 of 30
- nested: 19 of 19 / 27 of 30
- nested exact: 19 of 19 / 27 of 30

Read with the plan document (`docs/plans/2026-10-12-php-virtual-capstone-retarget.md`, runs 8 to 14):
the system allocator's catches are a strict subset of the nested arms'; CRASH-084 is out of the
denominator because this libc's putenv never creates the dangling reference the defect needs (checked
by a probe whose transcript names its image and whose hash the record carries); CRASH-075 is `infra` on
every arm (the 40 s watchdog; the plan document records a separate 600 s run of it, not archived here,
as a catch at `php_array_walk`); CRASH-092 is `infra` on the nested arms and `not-reached` on stock and
the system allocator (it hangs, or completes short of its defect, on uninitialised stack contents).
The native population and the system allocator's column disagree on 22 of the 49 cases: 13 asan-blind
cases the system allocator catches, and 9 temporal-asan-visible cases it does not read (8 `setup-fault`,
1 `unattributed`), which are not measured misses and not a second reading of where the block's release
went. The native cohort is a selection by one detector build, not a prediction of this column.

## Not here

The 59 default-build cases outside both populations: 30 spatial defects ASan sees unaided
(a free column for exact-bounds mallocng, every arm), 13 NULL dereferences and 11 type
confusions (controls, not capability claims), 5 more temporal. The 15 cases that need the
`fullext` build. The 43 wrong-value cases (`LOGIC-*`, `BIND-*`), whose oracle is a value,
not a fault.
