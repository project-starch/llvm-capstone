# QEMU references for the whole-corpus pass on bitstream 2 (caplifive_vp2c_dcc6daf6e)

The board lane runs every arm the batch pass covered on bitstream 2, beside the batch column. These are the arms'
QEMU references, made the way the batch references were (capstone/tests/batch-board-refs/README.md), with the kit
swapped.

## The twin, re-checked against the sealed hash (2026-10-24)

The QEMU twin is pinned to capstone-ariane d44c9441b (capstone-qemu `virtual/flat-index-18`, INDEX18-GATE.md).
Bitstream 2 sealed from dcc6daf6e, which descends from it. Between the two:
- `capstone_rev_node.anvil` and `capstone_unit.anvilh` change only comments: the 16383 → 4095 and 14 → 12-bit
  wording, now that the record fields are 12-bit.
- `ex_stage.sv`, `load_unit.sv` and `cva6.sv` cut the load unit's LDC clear ring and remove the tracer.
- The one deliberate behaviour change is R-61's fix. R-61 is a silicon defect (a refused LDC left the load unit's
  store-permission flag at 0) that QEMU never had.
- No record field, index width or id format moved, so the twin stands.

## The kits

- **qkit7-b2twin3-q8:** QEMU 708bbff5, module 33254735 (18-bit) and launcher fad465ce (8,192 mappings, the board
  kit's), with the batch kit's firmware and images.
  - It was made by hardlinking qkit7-b2twin2-q8, so its wrapper and applied.log were that kit's own. The binaries
    were identical, so no reading moved.
  - Since 2026-10-24 it has its own wrapper and log.
- **qkit7-b2twin3-q8-nr:** the same kit with the norandmaps wrapper, made as qkit7-batch2-nr is from qkit7-batch2.

## Pre-registration: the 25 pass arms, lt0, lt1, pg-pools and sql0 (2026-10-24, before the runs)

**The script:** `batch-refs-b2/run-pass-b2.sh`. It is batch-refs/run-pass-r60.sh (ebd2406b) with only these changes:
- the kits, as above;
- lt0 and lt1 on their application-mremap relinks, as full build directories (`t3-mremap/cpython-lt{0,1}-mremap-build`:
  the batch build with the image (lt1 8bcc40cf, lt0 7b871db7), the relinked SDK's runtime archive (34da9d0f) and the
  manifest replaced);
- every arm's cells compared with its batch reference, `batch-refs-r60/ref-batch/<tag>`, on verdict, attribution,
  cause, symbol and offset.

The runners, trees and arguments are the batch pass's. wt-gate's runner files are unchanged since the batch pass
(run-virtual-cases.py 50f10fa5, tier3-appvm-arm.sh 11967b37), and wt-q15cw, wt-cpyrec and wt-sdkgate are at their
stamped commits and clean.

**Predicted:**
- **The 25 pass arms, pg-pools and sql0:** every cell equals its batch reference. Bitstream 2 changes the index
  width, and no cell of these arms was a record-capacity stop (the coverage file's census finds all ten in lt1). The
  launcher change (512 → 8,192 mappings) moves no recorded reading. Any cell that moves is listed with its cause.
- **lt1:**
  - **The ten capacity cells (02, 10, 13, 18, 19, 23, 27, 28, 30, 31)** reach the sites the one-call-per-boot twin
    runs read (corpus-cell-probes):
    - 02: cause 28 in memcpy;
    - 10: l_divmod;
    - 13: unicode_fromformat_write_str;
    - 18: mmap_ass_subscript;
    - 19: fnv;
    - 23: memchr;
    - 27: slot_nb_add;
    - 28: _PyEval_Vector;
    - 30: PyObject_RichCompare;
    - 31: odict_copy.
    The verdict is whatever the judge makes of that fault against each case's fault_sites; the class and site are
    what is predicted.
  - **The other 22 cells** equal the batch reference.
- **lt0:**
  - **Case 18:** reads cause 28 in mmap_ass_subscript. In the batch reference it was MISSED but not reached
    (OSError 38).
  - **Case 13:** stays NO-READING:infra. This runner (wt-cpyrec dc606bd1) predates the MemoryError declaration
    (dev 3e28a2ff3d8e). The cell is MISSED by the lead's decision, which the declared runner reads
    (cpython/allocator-boundary-repros/results/2026-10-22-batch-declared).
  - **The other 30 cells** equal the batch reference.

The remaining arms follow, each pre-registered here before it runs: mruby stock and Sublet (smoke-fit), perl stock
and Sublet (full and trimmed), xlang, capi, plane, and Juliet temporal and spatial.

**The first launch was void** (2026-10-24 00:58, outputs kept in batch-refs-b2/failed-launch-1). The batch pass's
script took `CAPSTONE_QEMU_LOCK` from its launching shell. Launched without it, the domain lock became
`/qemu-domain-<SET>.lock`, which is not writable, and every boot failed at once (NO-READING:infra). The script now
sets the variable itself. The relaunch was at 00:59:41; the stamps above are from it.

## Pre-registration: the remaining arms (2026-10-24, before their runs)

**The script:** `batch-refs-b2/run-rest-b2.sh`, chained behind the pass. Each arm is its batch reference's own
invocation with the kit swapped to qkit7-b2twin3-q8, then `batch-refs-b2/cmp-any.py` against that batch reference.
The compare was tested before any boot: identical inputs read differ 0, real pairs read their differences (mruby
stock vs Sublet differ 4, capi run 1 vs run 3 differ 1), and a missing side reads NO DATA with exit 2.

| arm | batch reference compared | runner |
|---|---|---|
| mruby stock | batch-refs-r60/tick/mruby-stock | wt-ref 9db9f5f7 |
| mruby Sublet (smoke-fit) | smoke-fit/sublet | wt-ref 9db9f5f7 |
| perl stock and Sublet, trimmed | batch-refs-r60/batch/perl-{stock,sublet}-trim | wt-ref2 aabfa668 |
| capi | capi-repros results/2026-10-22-virtual-run3 | 6981dee6 |
| xlang repro and lua-cdp | xlang/{repro,lua-cdp}/results/2026-10-22-virtual-run2 | 6fcf3b75 |
| plane, both arms | plane-vp2/batch2-<arm> | wt-plane 1de34fdf |
| Juliet temporal | juliet-r60/judge-batch2 | wt-juliet run.py 4b663651 |
| Juliet spatial sample | juliet-sp/batch2 | wt-juliet run.py 4b663651 |

**Predicted:**
- **Every cell equals its batch reference.**
  - None of these arms had a record-capacity stop on the batch kit; mruby Sublet runs smoke-fit, which fits either
    table.
  - The launcher's mapping count moves no recorded reading.
- **Image hashes** for capi, xlang and plane equal the references', because the builds are path-independent.
- **Any moved cell** is listed with its cause.

## pg-pools on the trimmed fixture (pre-registered 2026-10-24, before its run)

The board runs pg-pools with the trimmed fixture and share, and was graded on the batch bitstream against pgp-trim/bundle
(capstone/tests/batch-board-refs/README.md). ref-b2/pg-pools ran the untrimmed fixture, as batch-refs-r60's pg-pools
did.
- **This run:** pgp-trim/run.sh's recipe with the kit swapped to qkit7-b2twin3-q8, in batch-refs-b2/pgp-trim.
  - runner run-arm.py 7ce203f9 from wt-sdkgate 03b1ecfcd33b (clean), the same hash pgp-trim's own tree carried;
  - image dc2a1356;
  - fixture pgfixture-trim/cluster and share pgfixture/install/share/postgresql.
- **Predicted:** 9/9 equal to pgp-trim/bundle on verdict, reason, cause and function, with fixture record
  956173e054762e91.

## lt0 09's site move: kit or image? (pre-registered 2026-10-24, before its runs)

lt0 on bitstream 2's twin read 30 cells as the batch reference and 2 moved:
- 18, as pre-registered;
- 09 (gh-140802), not predicted. It is CAUGHT on both sides, with cause 24, attributed by its control, but the
  fault's function moved from `_PyObject_GetMethod` to `dict_traverse`.

The run changed two things: the kit (18-bit) and the image (the mremap relink). One kit/image pair cannot say which
moved the site.

**The 2×2:**
- 09's case and control, one boot each (b2-pred/run-pred.sh);
- image: the batch image 23a2f0cb (session `lt0-09-batch`) or the mremap relink 7b871db7 (`lt0-09-mremap`);
- kit: qkit7-batch2-nr or qkit7-b2twin3-q8-nr.

**What decides it:** the case call's fault function in each of the four cells.
- If it follows the image (dict_traverse with 7b871db7 on both kits, _PyObject_GetMethod with 23a2f0cb on both),
  the relink moved it: the relink shifts code and data placement, and the defect reads garbage past an object.
- If it follows the kit, the 18-bit table did.
- Any other pattern (for example single-call boots differing from the session boots) is reported as read.

**My expectation, stated as such:** it follows the image. The relink is the change that moves memory contents; the
index width does not alter what an object's neighbouring bytes hold.

**Result** (2026-10-24 03:56-04:02, one boot per image per kit): **it follows the image, as expected.**

| image | qkit7-batch2-nr | qkit7-b2twin3-q8-nr |
|---|---|---|
| batch 23a2f0cb | cause 24 in `_PyObject_GetMethod` | cause 24 in `_PyObject_GetMethod` |
| mremap 7b871db7 | cause 24 in `dict_traverse` | cause 24 in `dict_traverse` |

- The control completes (exit 0) in all four.
- The relink moves 09's fault site, and the 18-bit kit moves nothing.
- 09 reads CAUGHT either way, so on bitstream 2 the board's 09 should read `dict_traverse`, as ref-b2/lt0 does.
- (run-pred.sh's own parser printed NO DATA for these two-call sessions, as its one-call guard intends. The readings
  above are read from each run's readings.json.)

## Result: the pass (2026-10-24 00:59-04:07; batch-refs-b2/PASS-A, PASS-B)

Every arm exited rc 0, and every stamp names QEMU 708bbff5, module 33254735 and launcher fad465ce.

| arm | cells | moved against the batch reference |
|---|---:|---|
| pg-pools (untrimmed), sql0 | 9, 9 | 0, 0 |
| sq0, sq1q15 | 33, 33 | 0, 0 |
| wm0, wm1q15 | 23, 23 | 0, 0 |
| wph, wpt | 12, 10 | 0, 0 |
| fph, fcs, fpt, fca, fso | 25, 13, 13, 13, 9 | 0 each |
| pym0, pym1 | 20, 20 | 0, 0 |
| pgm0, pgm1, pgm1q15, pgc | 5, 5, 5, 5 | 0 each |
| mph, mpt | 9, 3 | 0, 0 |
| pb9, pb10, pb12, pb13q15 | 4, 4, 9, 9 | 0 each |
| lt1 | 32 | 10: exactly the ten pre-registered capacity cells, each at its predicted site, now CAUGHT |
| lt0 | 32 | 2: 18 as pre-registered (cause 28 in mmap_ass_subscript, CAUGHT); 09's site, explained above by the image |

- **Held:** every pre-registered reading.
- **Not predicted:** one site move, lt0 09. It was traced to the mremap relink by a 2×2. Its verdict is unchanged.
- **Also held:** pg-pools on the trimmed fixture, 9/9 equal to pgp-trim (batch-refs-b2/pgp-trim, fixture
  956173e054762e91).
