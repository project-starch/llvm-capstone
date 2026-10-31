# SQLite engine-repros on silicon: three arms in one boot, every cell predicted first

**For the board lane.** This asks for one batched board session on `sqlite/engine-repros`, and it
is pre-registered: every arm's reading is written down below *before* the boot, including the two
that must NOT move. Nothing here needs a reflash. The corpus side is done and on `sqlite-rerun-v1`
-- `check-corpus.py` CLEAN, `derive-verdicts.py --check` current, `test_verdicts.py` 29/29 -- so
what is owed is board time and nothing else.

## Why

Two things changed under the board's feet since 2026-10-18/20, and both change what a board run of
this corpus means.

1. **Four triggers were corrected and one case withdrawn (2026-10-29/30).** Cases 02, 11 and 13
   each had a trigger that did not reach the defect it named; each now has a replaced trigger and a
   reading to show it. Case 08's could not be repaired and it is `withdrawn`, by the rule 01 and 09
   already carried. The denominator is now **27 counted cases = 19 temporal + 8 spatial**, out of 33
   directories (01, 04, 08, 09, 18 `withdrawn`; 25 `duplicate_of` 00).
2. **A catch is now the mechanism rejecting the access, not any fault.** `tools/arms.json` gives the
   three memsys5 configurations a `catch_causes` of `[25, 28]`. This was applied to the existing
   board bundles by **re-fold, not re-run** (`derive-verdicts.py --refold`, which refuses to alter
   an Observation), so **no board time is owed for it** -- see
   `docs/ref/sqlite-engine-repros-catch-causes-rescore.md`, which names every re-scored cell.

## What is already on the board, and what is stale

| arm | bundle | covers | state |
|---|---|---|---|
| `vp2-malloc` (base, nested) | `results/2026-10-18-vp2-rdx0/vp2-malloc` | 33 cases, merged from six `sq0` subsets | pre-correction triggers; re-folded to 0 CAUGHT |
| `vp2-nested-pools` (Sublet, nested) | `results/2026-10-20-vp2c-treeA/vp2-nested-pools` | cases **18-32** on `caplifive_vp2c_8522d72fe` | pre-correction triggers |
| `vp2-nested-pools` (Sublet, nested) | `results/2026-10-20-vp2c-treeB/vp2-nested-pools` | cases **00-17** on `caplifive_vp2c_8de3c48cf` | pre-correction triggers |
| non-nested on silicon | — | — | **never run on the board** |

So the protected arm's silicon coverage is two half-corpora on two different bitstreams, which
`merge-subset-bundles.py` correctly refuses to combine, and both halves ran the triggers that have
since been corrected. Tree B's 6 MISSED over cases 00-17 is what a pre-correction trigger looks
like on an arm that now reads 19 of 19 on QEMU.

## The ask: three arms, one bitstream, one boot sequence

All three on **one** bitstream, so the three bundles are comparable and `merge-subset-bundles.py`
will accept each arm's subsets. Order them cheap-first, as the batching rule says.

| # | arm | configuration | what it measures | QEMU reference bundle |
|---|---|---|---|---|
| 1 | `vp2-nested-pools` | `virtual-sqlite-memsys5-pools` | Sublet over memsys5: the protected nested arm | `results/2026-10-29-virtual/virtual-nested-pools` |
| 2 | `vp2-malloc` | `virtual-sqlite-memsys5` | the same arena with no lifetimes below it: the base column | `results/2026-10-29-virtual/virtual-malloc` |
| 3 | `vp2-platform` | `virtual-mallocng` | **new arm**: the same defects with memsys5 left out of the build | `results/2026-10-29-virtual/virtual-platform` |

`vp2-platform` follows `vp2-malloc`'s precedent exactly: a board arm name of its own so both
bundles stay current, and the virtual configuration, because the configuration is what is measured
and the platform is what differs (`SCHEMA.md`, the `vp2-malloc` row).

## Pre-registered readings

Over the **27 counted** cases. Each arm's 33-row bundle also carries the six out-of-denominator
cases; score cell by cell against the QEMU bundle named above, as the q15 runs did with
`subset-bundle.py` and `compare-pilot.py`.

| arm | temporal (n=19) | spatial (n=8) | bundle tally over all 33 | QEMU reads |
|---|---|---|---|---|
| `vp2-nested-pools` | **19 / 19** | **7 / 8** (31 by R-60) | CAUGHT 28, MISSED 5 | 19/19 and 8/8 |
| `vp2-malloc` | **0** / 17 (+2 no reading) | **0 / 8** | 0 CAUGHT | 0/15 (+4) and 0/8 |
| `vp2-platform` | **17 / 17** (+2 out of denominator) | **5 / 8** (31 by R-60) | CAUGHT 24, MISSED 7, NO-READING 2 | 17/17 (+2) and 6/8 |

**`vp2-malloc`'s no-reading count is the BOARD's, not QEMU's, and the difference is already
measured.** QEMU's `virtual-malloc` has four cells without a reading and the board bundle has two:
case 11 reads `NO-READING:inconclusive` on QEMU because memsys5 recycling inside a 1 MiB arena never
lets `fts3EvalGatherStats` finish, and it returns on the board. The prediction above is the existing
board bundle's split, which is what this re-run must reproduce.

**Spatial 7 / 8 on `vp2-nested-pools`, not 8 / 8, and that is not a discrepancy.** Case 31
(`31_fz10_dbstat_decodepage_overflow`) is MISSED on vp2 and CAUGHT on QEMU **by R-60**: the `lbu` in
`statDecodePage` goes through a capability whose cursor was moved 50,976 bytes past the end of a
20,160-byte buffer, outside its representable window, and vp2 checks it against the bounds decoded
at that cursor -- the true bounds shifted by one 64 KiB span, which contain it. Already measured on
both bitstreams and on three boots, with the shifted bounds printed by the diag31 boot
(`ISSUES.md` R-60, and the per-platform note in `results/2026-10-20-vp2c-treeA/README.md`). If the
bitstream chosen for this run carries an R-60 fix, the prediction for spatial becomes 8 / 8 and
case 31 is the cell that says so.

**On `vp2-platform` the same prediction for case 31 is an INFERENCE, not a measurement.** R-60's
mechanism is a property of the access and of the compressed-bounds encoding, not of the allocator
layer: the object is bounded to the same 20,160-byte request whether it comes from memsys5 under
Sublet or from mallocng directly, and the cursor goes the same 50,976 bytes past its end. So case
31 ought to read MISSED on this arm too, which is why spatial is predicted 5 / 8. It has never been
measured there -- the non-nested arm has never run on the board -- so a CAUGHT is informative rather
than a contradiction, and would say the window depends on something the two arms do not share.

### The two readings that must NOT move

- **`vp2-malloc` reads 0 CAUGHT on both axes.** It already does, after the re-fold, and the
  mechanism rule is what makes it 0 -- not the attribution and not the trigger corrections. A
  corrected trigger that produced a *catch* on this arm would contradict the whole base column, so
  this arm re-running is the control on the corrections, not a formality.
- **Every case's PASS/FAULT outcome repeats.** The corrections changed what cases 02, 11 and 13 do;
  they changed nothing about the other 30, and a changed outcome there is a finding about the
  platform or the kit, not about this corpus.

## What this does NOT need

- **No reflash.** Any bitstream that already runs this corpus will do; name it in each bundle's
  `platform.qemu` as `board:<bitstream>` the way `vp2-malloc` does.
- **No re-run for the `catch_causes` re-score.** That was a re-fold of stored Observations.
- **No new images for arms 1 and 2** beyond rebuilding from the corrected cases: the amalgamation is
  unchanged (`amalgamation_sha256` `7fc92e19a3af61f3961b83447a91f1e1dec06646e343037dfd96a397167700b2`,
  reproducible from stock 3.22.0 through `ports/sqlite/adapt-sqlite-322.sh`), so only the three
  corrected cases' own `case.c` differ.
- **Arm 3 needs its own images**: the non-nested build is `build-virtual.py` *without* `--memsys5`
  (`build.json` arm `platform`), which is what `virtual-platform` runs on QEMU.

## What we owe before the bundles can be declared

`vp2-nested-pools` and `vp2-platform` are not yet arms of this corpus: each needs an entry in
`required_arms` and `ARM_ORACLES` (`tools/check-corpus.py`), an `arm_configurations` mapping and a
`verdict_bundles` path. That is a corpus-side change, it is ours, and it lands when the bundles
arrive -- the arms are deliberately NOT declared in advance, because a declared arm with no bundle
fails the gate and a declared arm with a *stale* bundle is worse.

## How to run it

This does not invent a procedure. The precedent is the q15 arm, which ran this exact corpus in six
`--only` subsets with both memsys5 controls in every boot:
`tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-q15-board{1,2}.sh`, with `Q15_SUBSETS` and
`EXPECT_IMAGES` cut to each subset's cases plus the controls, scored at once per boot by
`subset-bundle.py` then `score-arm.sh`, and merged by `merge-subset-bundles.py`. Records land in
that kit's `records/`. Two things from that run's own experience are worth keeping:

- **the payload cap** that made six subsets necessary rather than one boot per arm;
- **the image-list gate** (`EXPECT_IMAGES`, each subset's cases plus exactly two controls), which is
  what catches a subset booted against the wrong image set.

The console URL and token stay out of this file and out of every committed record; use
`<FPGA-CONSOLE-URL>` in anything written down, and name the board with `FPGA_BOARD=1|2`.

## Stop conditions

- **A control that does not behave stops the arm, not just the cell.** Both memsys5 controls
  (`uaf-mem5`, `bounds-mem5`) must appear in every subset; a MISSED with absent controls is
  `NO-READING:control-failed` by the judge, and a whole subset of them means the boot measured
  nothing.
- **A `vp2-malloc` CAUGHT stops the run and comes back to us.** It contradicts a column this
  review publishes as 0.
- **A changed PASS/FAULT outcome on any case other than 02, 11 or 13** likewise.
- Anything that reads as R-60, R-61 or R-64 is already characterised; cite the issue rather than
  scoring it as a corpus discrepancy.

## Provenance

Pre-registered by this file on branch `sqlite-rerun-v1`; cite its commit in whatever the run lands
under. The corpus state it predicts from: `check-corpus.py` CLEAN over 30 corpora and 399 cases,
and `results/verdicts.tsv` is the cell-by-cell table for all seven arms.
