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

**Result** (2026-10-23 03:50-03:59, QEMU domain B; batch-refs-r60/run-perl-trim.sh): **as predicted.**
- **perl-stock, trimmed vs full:** 11/11 equal, differ 0 (fault 6, completed 4, exit255 1).
- **perl-sublet, trimmed vs full:** 11/11 equal, differ 0 (fault 8, completed 3).
- Both arms' controls (eval, shim) behave.
- The stamps show runner aabfa668, tree 15ced322c824 clean, qemu ecaf50bb, module a05c198b, vexec a405840d,
  closure 2a472e21, and the images as above.
- The trimmed references are batch-refs-r60/batch/perl-{stock,sublet}-trim/out, for the board's trimmed staging.

## Juliet temporal on the batch kit (pre-registered 2026-10-23, before the build and runs)

**Why:** every other corpus arm has a batch-bitstream board reading. Juliet's references exist only for the tree
kits, built with the older cw SDK (juliet-vp2/judge-{tB-mimic,nf2}, sdk.json 64bc9086).

**The build:** juliet-vp2/build2.sh's recipe and selection, with only the SDK changed.
- Recipe tree wt-corpus b3cc8cb4be5a (clean); `select.py` manifest 95ceac7daa991165, 366 cases.
- `build.py`, census and `--seeded`, on a writable copy of the r60 SDK t3-freeze-nb-nomovc-cw-r60 (sdk.json f5145cbb).

**The judging:** juliet-vp2/judge.sh's recipe with kit qkit7-batch2.
- `run.py` 4b663651 and `probe.py` d6200b39 from wt-juliet a772f0112ae3 (clean).
- `judge.py` 93aa5ad9 from wt-corpus, and the same juliet-timeout helper bb65b77a.

**Predicted:** 366 CAUGHT (228 CWE-415, 138 CWE-416), every good run CORRECT, no MISSED row. Every case equals
judge-tB-mimic's temporal-verdicts.tsv on verdict, good, cause, site and strict. A case that differs is listed with
both readings.

## Juliet spatial sample on the batch kit (pre-registered 2026-10-23, before its runs)

**The build:** the sample's existing census build, already made with the r60 SDK (juliet-sp/build-census, sdk.json
f5145cbb; manifest-spatial-sample.json, 755 cases). It is not rebuilt.

**The judging:** juliet-sp/run-judge.sh's recipe with kit qkit7-batch2 in place of qkit7-tB-mimic: `run.py`,
`probe.py` (wt-juliet a772f0112ae3), `judge.py` (wt-corpus b3cc8cb4be5a). There is no seeded build; the sample has no
rand() flow.

**Predicted:** every case equals juliet-sp/tB-mimic's spatial-verdicts.tsv on verdict, good, cause, site and strict
(716 CAUGHT, 22 MISSED, 9 NOT-TRIGGERED:sizeof-ptr, 8 intra-object NO-READING). The exception allowed in advance is
the CWE-126 CWE-170 family: whether its over-read runs on depends on whether a NUL follows the copy. Three of its rows
moved in both directions between the 2026-10-19 record and tB-mimic (juliet-sp/RESULT), so a move there is reported
as value-dependent, not as a platform difference. Any other moved case is listed and counts against the prediction.

**Results, both as predicted** (2026-10-23, qkit7-batch2: QEMU ecaf50bb, module a05c198b, vexec a405840d; QEMU domain A):
- **Temporal** (04:20-04:28; juliet-r60/RESULT):
  - built census 734 and seeded 55 images, as on the old SDK;
  - 366 CAUGHT (228 CWE-415, 138 CWE-416), every good run CORRECT;
  - 366/366 equal to judge-tB-mimic on verdict, good, cause, site and strict, differ 0.
  - Reference: juliet-r60/judge-batch2/results.
- **Spatial sample** (04:28-04:42; juliet-sp/RESULT-batch2):
  - 24 batches in 793.6 s;
  - 755/755 equal to juliet-sp/tB-mimic, differ 0, including the value-dependent CWE-170 rows.
  - Reference: juliet-sp/batch2/results.
