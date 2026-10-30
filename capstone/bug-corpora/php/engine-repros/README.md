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
| `asan-blind` | 48 | ASan reports it only with the allocator patched; as shipped the run reads OK, or for a temporal defect ends in PHP's own fatal error with no report (CRASH-062, -070). By CWE: 32 use-after-free, 10 out-of-bounds reads and 2 writes, 2 stack-address escapes, 1 free of non-heap memory, 1 uninitialised read; 7 of the 48 crash natively | selected by measurement, not by mechanism: what each arm sees is in the arms' records. The first run has the nested layer arms (`virtual-nested-pools`, `virtual-nested-exact`) at 26 and 30, the outer allocator alone (`virtual-malloc`) at 13 -- 8 use-after-frees whose block does reach `free()` plus 5 bounds catches the platform gives every arm |
| `temporal-asan-visible` | 22 | a temporal defect (19 CWE-416, 2 CWE-590, 1 CWE-415) that ASan reports with the allocator as shipped (the `stockalloc` run) AND that the patched build reports as a use-after-free or an invalid free: the block reached `free()`; 20 of the 22 crash natively, so the judge's default denominator holds two of them and the `--score-oracle-crashes` sensitivity scoring reads the rest | the outer allocator by itself, `virtual-malloc` (musl mallocng, a lifetime per block), for a block that reaches `free()` and is not parked in `AG(cache)`; the nested layer arms for the parked ones. Which is which is in the arms' records |

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

**One PHP corpus (2026-10-28).** A sibling corpus built from the same source by another lane (56 temporal cases) is merged into this one: 47 of its cases were already here, and five more that run on the default build are admitted by the widened temporal rule and numbered 66 to 70 after the first 65, so no number moved -- CRASH-012 and -148 (a free of the globals symbol table, CWE-590), CRASH-161 (a double free, CWE-415), CRASH-062 and -070 (use-after-free whose as-shipped run ends in PHP's own fatal error). Its four fullext cases (CRASH-125, -129, -131, -132: mbstring, pcre) need a build this corpus does not make and are not here. 70 cases, 55 of them temporal (51 CWE-416, 3 CWE-590, 1 CWE-415).

The first build gave 65 cases (46 + 19) where the collaborator's instrument measured 60 (31 + 29):
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
  --population asan-blind --image php-real-nested.dom --probe probe-nested.dom --llvm-bin "$CAPSTONE_LLVM_BIN" \
  --qemu ... --images ... --adapter ... --out results/<stamp>/asan-blind
```

One guest boot per arm, every case its own `capstone-vexec` process. The runner drives four
configurations of one nested image (`tools/arms.json` `php-real-stock|block|object|exact`,
`PHP_CAP_MODE` 0/4/6/7) and writes one bundle per arm under the framework's names
(`virtual-malloc` = block, `virtual-nested-pools` = object, `virtual-nested-exact` = exact);
the stock run is the judge's control for `reached` and keeps its own name. A run whose
transcript shows a defaulted arm (`ZEND_CAP_MODE_DEFAULTED`) or another arm's mode is refused.
`corpus.json`'s `verdict_bundles` names the current bundles, `derive-verdicts.py` writes the
verdicts into each `case.json`, and `check-corpus.py` re-judges them.

The fourth arm runs on a different platform and so has its own runner, which reports into the
same judge and writes the same bundle contract:

```sh
bash capstone/ports/php/cheribsd/build.sh                 # once: the purecap interpreter
python3 capstone/bug-corpora/php/tools/run-php-corpus.py \
  --corpus capstone/bug-corpora/php/engine-repros \
  --arm cheribsd-revocation --eager \
  --scratch /tmp/capstone/php-cheribsd/stage \
  --out capstone/bug-corpora/php/engine-repros/results/<stamp>
```

Three boots, not one: the platform default is the only configuration a verdict comes from, the
same binary with revocation off is the pair that attributes a fault (a purecap `SIGPROT` carries
no symbol, so the pair is what ties it to revocation), and `--eager` adds a third at
revoke-on-every-free whose only job is to tell a quarantined-but-never-reissued chunk apart from
a genuine miss. Both scorings are written from those same boots -- the default one under
`<arm>/`, the sensitivity one under `sensitivity/<arm>/` -- because the two readings have to
travel together and re-booting for the second would cost three more boots.

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

## Pre-registration for the `cheribsd-revocation` arm, before it ran (2026-10-30)

A fourth arm, declared in `corpus.json` and written into every case by `tools/build-cases.py`:
the same triggers on **stock CheriBSD purecap**, with no nested layer at all, so the only
mechanism is the platform's -- `malloc` bounds each object, `free` quarantines it, and a later
sweep revokes. The port is `capstone/ports/php/cheribsd` (`build.sh` cross-builds the real
interpreter; `run-cheribsd-cases.py` boots the guest), driven by
`bug-corpora/php/tools/run-php-corpus.py`.

Written before this arm ran over these 70 cases. The prior is not a guess: the same vehicle has
already measured 56 of these defects in `../plain-temporal-repros` and `../zend-alloc-repros`,
and `capstone/tests/cheri-php-blind` has measured the mechanism in C rows. Both are cited per
group, and the case numbers are this corpus's.

**The vehicles are not identical, which is why each bundle now records its own.** That row-model
measurement ran a 7.2.0 CHERI-QEMU fork with `-cpu rv64,sstc=false`; this port runs 7.1.0, which
has no such property. `run-cheribsd-cases.py` reads the QEMU and CHERI-LLVM versions, the
firmware and the image's size and mtime from the tools at run time into the record, so a reader
can tell whether two tables share a vehicle instead of assuming it.

| group | cases in the default denominator | prediction at the platform default | what it rests on |
|---|---|---|---|
| temporal, release stops in `AG(cache)` | 20: 01 02 04 06 07 08 09 10 12 16 18 20 22 26 33 35 37 42 43 46 | **0 CAUGHT.** Some abort in CheriBSD's allocator consistency check (`SIGABRT`, rc 134), which is recorded and never scored as a capability catch | `../zend-alloc-repros` measured 30 such defects on this vehicle: 0 protected at the default. `cheri-php-blind` caught 0 of its 4 cache-hidden UAF rows in the stock arm |
| temporal, release reaches system `malloc` | 6: 21 38 55 58 67 68 | **0-2 CAUGHT**, and at most a minority | `../plain-temporal-repros`: of 26, three faulted `SIGPROT` at the default and only ONE of those completes with revocation off, so only one was revocation's |
| temporal, in neither of our corpora | 3: 25 30 41 | no prediction beyond the mechanism above | nothing measured |
| spatial (rounding-slack overflow) | 12: 03 05 17 19 27 28 29 31 32 34 36 40 | **MISSED.** A CAUGHT here means the over-read left the block, which is a finding about that case and not about CHERI | `cheri-php-blind`'s `z1_bounds_probe` read the capability directly: the `emalloc`'d pointer carries the whole block's base and length, and removing PHP's rounding does not help because `malloc` rounds by the same amount -- 6 readable bytes past the object at n=26, 232 at n=1048 |
| stack-address-escape (CWE-562) | 2: 13 14 | **MISSED** | `cheri-php-blind`'s `c1_stack_after_return` is MISS in all three configurations and both arms: a returned stack capability stays tagged and in bounds, and CheriBSD does not revoke stack capabilities on return |

So **0 to 2 CAUGHT of the 43** in the default denominator, and a headline at or near zero. A
count above that is a surprise and is to be read as one.

**The every-free (eager) configuration is context and may never decide a verdict.** The
prediction there is specific, because it is the one place our own 56-case run found something
the default cannot see: a group of the cache-absorbed cases above will fault under every-free
while the default and the revocation-off boots both complete, because the capability ABI pushes
the payload past `MAX_CACHED_MEMORY`'s 80-byte ceiling and the block reaches a real `free()`
after all. Seven of thirty did exactly that in `../zend-alloc-repros`, every one on a 192-byte
block -- a 48-byte purecap `zend_mem_header` over a 144-byte payload, `REAL_SIZE >> 3` = 18
against a ceiling of 11 -- as tag violations at the defect's own dereference site. Those eager
faults are predicted to land **inside the script body**, not in request shutdown.

**Controls, without which the run is void.** `bounds-malloc` must fault in every boot, or bounds
are not enforced on this vehicle and no MISS means anything. The revocation-off twin of each boot
is run as the pair that attributes a fault: a fault that appears with revocation off too is not
revocation's, and is recorded as such. The eager boot must print the request-shutdown marker for
at least one case, or a fault's placement reads `unplaced` rather than "inside the script body".

**The 27 cases whose pristine record is a `SIGNAL`** read `out-of-denominator` under the default
scoring, whatever this arm did -- the same gate, and the same test
(`pristine_outcome.startswith("SIGNAL:")`), that `run-realphp.py` applies. The sensitivity
reading is written beside it in `sensitivity/` from the SAME boots, so nothing differs between
the two bundles but the scoring.

## Results

Seven records, each a run of the whole corpus on the virtual profile, one bundle per arm with the two
populations merged (`tools/merge-population-bundles.py`), every table computed from the bundles by
`tools/result-tables.py` and written into that record's README, never typed:

- `results/2026-10-27-run7/`: lane B's measurement (the corpus at `672a0deffaa2`, 65 cases).
- `results/2026-10-28-run10/`: lane C's measurement (the nested layer on the libc's reclaim, `8569fb7194dc`): against run7 no CAUGHT changed.
- `results/2026-10-28-run14/` and `results/2026-10-28-run15/`: the follow-up lane's measurement (the 8 MiB-stack SDK with line tables for the musl objects, the launcher that records the frame chain and, from run15, names its image on exit, the putenv probe); run15's verdicts and reasons are cell-identical to run14's.
- `results/2026-10-28-run16/`: the merged corpus measured whole (70 cases): the first 65 exactly as in run15, the five merged cases measured for the first time.
- `results/2026-10-28-run17/`: the same 70 cases on the reduced nested layer, every cell identical to run16.
- `results/2026-10-28-run18/`: **the current record**, the same 70 cases under the renamed layer ("seam" is now "nested"), every cell identical to run17. `corpus.json`'s `verdict_bundles` names its arm bundles, `derive-verdicts.py` wrote their verdicts into every `case.json`, and `check-corpus.py` re-judges them.

Over the 55 temporal cases (51 CWE-416, 3 CWE-590, 1 CWE-415), from `results/2026-10-28-run18/README.md` (identical to run16's and run17's):

**Default scoring** (the native crashers out of the denominator):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 1 | 1 | 24 | infra 1, not-reached 2, out-of-denominator 27 |
| system allocator | 12 | 1 | 12 | infra 1, not-reached 2, out-of-denominator 27, unattributed 1 |
| nested | 25 | 1 | 0 | infra 2, out-of-denominator 27, unattributed 1 |
| nested exact | 25 | 1 | 0 | infra 2, out-of-denominator 27, unattributed 1 |

**Sensitivity scoring** (every case scored by fault site; reported beside the default, never instead):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 2 | 1 | 24 | infra 3, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 5 |
| system allocator | 25 | 1 | 12 | infra 2, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 4 |
| nested | 48 | 2 | 0 | infra 3, out-of-denominator 1, unattributed 3 |
| nested exact | 48 | 2 | 0 | infra 3, out-of-denominator 1, unattributed 3 |

Read with the plan document (`docs/plans/2026-10-12-php-virtual-capstone-retarget.md`, runs 8 to 16):
the system allocator's catches are a strict subset of the nested arms'; a catch "by chain" is attributed
through the call path alone (a fault in a libc copy, scan or write routine, the declared function reached
by walking its callers: CRASH-068 through ra on every arm, CRASH-057 through the frame chain on the
nested arms) and is the weakest attribution class, shown apart and never counted silently. CRASH-084 is
out of the denominator because this libc's putenv never creates the dangling reference the defect needs
(checked by a probe whose transcript names its image and whose hash the record carries). CRASH-012 and
-148 free a symbol table malloc never returned: the nested layer refuses the pointer on every arm, which the
judge reads as `unattributed` (their declared site is PHP's allocator). CRASH-075 and CRASH-161 are
`infra` on every arm (the 40 s watchdog; the plan document records a separate 600 s run of 075, not
archived here, as a catch at `php_array_walk`); CRASH-092 is `infra` on the nested arms and
`not-reached` on stock and the system allocator (it hangs, or completes short of its defect, on
uninitialised stack contents).

The native population and the system allocator's column disagree on 24 of the 51 CWE-416 cases: 15
asan-blind cases the system allocator catches, and 9 temporal-asan-visible cases it does not read (8
`setup-fault`, 1 `unattributed`), which are not measured misses and not a second reading of where the
block's release went. The native cohort is a selection by one detector build, not a prediction of this
column: the sibling corpus's own ASan measurement of the same 47 shared cases placed 16 of them on the
other side.

## Not here

The 59 default-build cases outside both populations: 30 spatial defects ASan sees unaided
(a free column for exact-bounds mallocng, every arm), 13 NULL dereferences and 11 type
confusions (controls, not capability claims), 5 more temporal. The 15 cases that need the
`fullext` build. The 43 wrong-value cases (`LOGIC-*`, `BIND-*`), whose oracle is a value,
not a fault.
