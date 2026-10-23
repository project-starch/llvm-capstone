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
