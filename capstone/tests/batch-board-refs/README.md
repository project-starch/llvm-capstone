# QEMU references the board lane asked for, for the batch bitstream (c721c1a2a)

Each one is run on qkit7-batch2: QEMU ecaf50bbd5ec44aa, module a05c198bfcc5f55c, vexec a405840d, with
movc_null_scalar OFF, because MOVC keeps on the batch bitstream (Q-04 (b)). Each prediction is written here and pushed
before its run.

## pg-pools on the trimmed fixture (pre-registered 2026-10-22, before its run)

The board's sql0 path uses `t3-new6/tmp/pgfixture-trim/cluster` (run-arm record 956173e054762e91) with
`--pg-share t3-new6/tmp/pgfixture/install/share/postgresql`, under the 26,000,000-byte declared-bake cap. The batch
reference `batch-refs-r60/ref-batch/pg-pools` ran on the UNTRIMMED cluster 173aa5b35c1b57b9, with no --pg-share and
runner 1f099612 (23ef40b664eb).

This run changes the fixture, the share and the runner. The runner becomes dev's 7ce203f9 (87324715e232), the one
sql0's board path requires. Everything else stays: image dc2a1356dce735e0 (t3-images-r60/q15/postgres-lt1),
`--arm virtual-pg-pools`, and llvm-bin-v6.

The prediction: all 9 cells read as the untrimmed reference, outcome, cause and function, and all 4 controls fault:
- CAUGHT 01, 02, 03, 04 and 06;
- MISSED 05 and 08;
- NO-READING:unattributed 07 and 09.
The precedent is sql0, whose trimmed record equalled the native16 record in 9 of 9 cells. Any moved cell is reported
as a difference of the fixture, not averaged away.

**Result** (2026-10-22 14:40-15:21, QEMU domain B; /mnt/hdd/<user>/capstone/pgp-trim):
- **The fixture:** the bundle's record is 956173e054762e91, the trimmed cluster sql0's board path uses.
- **The cells:** untrimmed → trimmed, 9/9 equal (outcome, reason, cause and function), MOVE 0. CAUGHT 01, 02, 03,
  04 and 06; MISSED 05 and 08; NO-READING:unattributed 07 and 09.
- **The controls:** all 4 fault (bounds-malloc, uaf-malloc, bounds-palloc, uaf-palloc).
- **The stamp** (pg-pools-trim.stamp) gives runner 7ce203f9, qemu ecaf50bb, module a05c198b, vexec a405840d and image
  dc2a1356.
  - Its SWITCHES line is the kit's, movc_null_scalar=OFF. All 5 lines logged in the run's window are identical. The
    kit log names no caller.
  - The first stamp took a line from another day: my time filter compared clock times as strings. That was corrected
    before this record.

Every pre-registered reading holds.

## perl-stock and perl-sublet with TRIMMED staging (pre-registered 2026-10-23, before the runs)

**Why:** the board runs perl with trimmed staging (`--lib-closure`) because of the payload cap, and the batch perl
references (batch-refs-r60/batch/perl-{stock,sublet}) ran FULL. So the board compares a different staging against
its own.

**The runs:** the batch references' own invocation (batch-refs/run-mp-r60.sh), with `--lib-closure` added:
- runner `perl/release-differential/runners/virtual/run-virtual.py` aabfa66858b2c35e, from wt-ref2 at 15ced322c824
  (clean);
- the closure `board/lib-closure.txt` 2a472e21452045e1, the one corpus-mp/run-refs-perl2.sh's tB-mimic trimmed
  references used;
- kit qkit7-batch2 and the r60 images: perl-stock 76f1141df548fd9a, perl-sublet bec05c860fb7e808 (runtime
  4a5e480c1680c87d), as the full references' stamps record.
- Each arm's trimmed matrix is compared cell by cell with its full one, on outcome, cause, function and
  in_fault_sites. The harness column carries addresses and is not compared.

**Predicted:** both arms read 11/11 equal, differ 0, as the same gate read on tB-mimic (corpus-mp/refs/REFS: perl-stock
and perl-sublet full vs trimmed, 11/11, differ 0). The full matrices hold perl-stock fault 6, completed 4, exit255 1,
and perl-sublet fault 8, completed 3.
