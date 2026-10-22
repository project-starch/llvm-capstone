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
