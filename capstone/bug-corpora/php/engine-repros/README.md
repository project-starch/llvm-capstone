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

## The first run

`results/2026-10-27-first-run/`: both populations on dev `0b0b3414dca5` with the patched port, one bundle per arm (the two populations merged by `tools/merge-population-bundles.py`, sources by hash). Its README has the tables; `derive-verdicts.py` wrote the verdicts into every `case.json` from those bundles, and `check-corpus.py` re-judges them.

## Not here

The 59 default-build cases outside both populations: 30 spatial defects ASan sees unaided
(a free column for exact-bounds mallocng, every arm), 13 NULL dereferences and 11 type
confusions (controls, not capability claims), 5 more temporal. The 15 cases that need the
`fullext` build. The 43 wrong-value cases (`LOGIC-*`, `BIND-*`), whose oracle is a value,
not a fault.
