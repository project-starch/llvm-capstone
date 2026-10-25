# Silicon readings on bitstream 2 dcc6daf6e: what b2-silicon-tally.tsv holds (board lane, 2026-10-25)

Same generator, same columns and same out-of-denominator rules as batch-silicon-tally.tsv: 38 rows, 1,625 cells.

## What changed from the batch bitstream (c721c1a2a)

Exactly two rows differ from batch-silicon-tally.tsv; the other 36 are identical in every count column.

- **cpython-lt1:** CAUGHT 17 → 27, and NO-READING 12 → 2 (infra 4 and unattributed 7 → unattributed 1).
  - These are the ten cells the 16-bit table stopped. On bitstream 2's 18-bit table each is CAUGHT at the 18-bit QEMU
    twin's site.
  - This row runs on the application-mremap relink 8bcc40cf, where the batch row ran 916e14f9.
- **cpython-lt0:** CAUGHT 15 → 16, MISSED 14 → 13.
  - Case 18 is CAUGHT, with cause 28 in mmap_ass_subscript.
  - This row runs on the mremap relink 7b871db7, where the batch row ran 23a2f0cb. The relink is what lets case 18 reach
    its site (the batch case-18 call read the same).

## Sources and verdicts

- Every row is read from the BOARD's own bundles, under `<mnt-hdd>/capstone/t3-run-b2-dcc6daf6e-b{1,2}/`, given in the
  source column.
- **cpython-lt1** was read by `tier3-appvm-salvage.sh` from the boot's own transcript. The boot completed, but its
  wrapper timed out. The salvage has a positive control (records/salvage-control-lt1.txt).
- **The Juliet rows** carry the bitstream-2 QEMU reference's counts (batch-refs-b2/rest/juliet-*), as the batch file
  did. judge.py cannot read CAUGHT on a board, so the board's cell-for-cell equality under juliet-obs-compare carries
  them:
  - temporal census 726/735, the 9 differences all rand-driven `_12` draws, scored from the seeded build, which is 56/56;
  - spatial 1513/1513.
- **Every row matches its bitstream-2 QEMU reference** (the compiler lane's 18-bit twin qkit7-b2twin3-q8, every
  reference equal to its batch reference except the lt0/lt1 cells above), except for the Q-31 cells. Those are the same
  verdict and site with cause 4/13 where QEMU reads 24: lt0 01, and sq0 cells 19, 20 and 22.
- The records are lane vp2-tier12 `records/b2bs-*.result-lines.txt`, on dev with this file's landing.

## Columns

- **nested** is the ARM: virtual-nested-pools and virtual-pg-pools are nested; virtual-malloc is plain.
- **No spatial/temporal column.** Use the corpus name: plain-temporal-*, pymalloc-repros and juliet-temporal are
  temporal; plain-heap-*, subobject, carved and juliet-spatial are spatial. allocator-boundary (cpython), sql-repros,
  capi and xlang mix the two.
- **NO-READING_by_reason** is the runner's reason field.

## Out of the denominator (verdict NO-READING, reason out-of-denominator), unchanged

- xlang repro: 11 mruby rows (lead decision (b)).
- capi: 7 rows:
  - rows 12, 13 and 14, null-deref (lead decision (c));
  - rows 17, 18 and 19, which exit 0 with nothing to catch;
  - row 15, which has no program.
- Count them out of n when computing catch rates.

## Release-differential arms (mruby, perl)

- These arms have no CAUGHT/MISSED verdict. Their "other" column gives the outcome per case, and each row equals its
  bitstream-2 QEMU reference cell for cell.
- mruby Sublet ran with smoke-fit.

## Repeatability

- Every row is ONE board reading of its arm on this bitstream (N=1).
- A second pass (N=2), graded cell for cell against these readings, is running on both boards. It is pre-registered at
  lane 60709f69bfc2.
