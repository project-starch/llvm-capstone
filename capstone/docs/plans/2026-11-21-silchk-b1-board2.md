# Bitstream-1 silicon-check kit on board 2: board lane run plan and an amendment to the resident tree

Status: written and pushed 2026-11-21, before the run. The kit and its grading are the rtl lane's (land/rtl-silchk-b1
144d71110532, `capstone/tests/rtl-smoke/silchk-b1-2026-11-21/PREREG.md`), unchanged. This file adds one amendment and
the procedure.

**Amendment: the resident bitstream is P1, not bitstream 1.**
- The runtime lane asked for the kit on board 2's resident P1 without a flash
  (`caplifive_…-pagedtbl_d83d3a633.bit`, sha256 dc411e36…). The train chain behind it then needs no reflash.
- Bitstream 1 (9e801ab5e) differs from P1 only by the R-76 ccsrrw term in csr_regfile.sv. That term refuses ccsrrw
  below M outside supervision.
- Read at the kit's sources (144d71110532): the only ccsrrw in any image is `va-runv.S:173`. It is in the M-mode
  monitor's setup, before the `mret` into S at :221. The S-mode code runs CSRUNV and `ecall`, never ccsrrw. No other
  test uses ccsrrw.
- So the term cannot change any of the seven vectors, and the PREREG's bitstream-1 vectors are the predictions on P1.
- The four fixes' files are byte-identical between P1 and bitstream 1 (the PREREG's own statement).
- The resident name is read back and recorded with the readings. A resident other than P1 STOPs the run before any load.

**Procedure** (`<mnt-hdd>/capstone/silchk-b2/run-silchk.sh`):
- Its stubbed dry runs, `dry1` (every rung) and `dry-neg` (a STOP branch), are kept beside it.
- It waits for the runtime lane's probe chain to end, then for board 2 to be free.
- It reads back the resident name, which must be P1's, and books board 2 for 60 min.
- It runs one image per JTAG load with `sup-bare-2026-10-03-ladder/run_sup_bare.py` from the kit's tree, in order:
  va-runv (the control) first, then alias-store, pcc-end-in, pcc-end-out, pcc-end, shrinkto, and niss-superpage last.
  Each run's timeout is 150 s.
- On every exit it unbooks and touches the train chain's marker (`perf/natstage/silchk-b2.DONE`), so the board does
  not idle.
- Every image's sha256 and `.rec` address equal the PREREG table, checked before the run.

**Grading:** the PREREG's rules, applied value by value to the UART's SR/SV lines.
- VOID unless va-runv reads SR 15 and its 15 values.
- R-68 is graded by its ratio keys, and each arm's paddr word must match.
- Readings are cited by image hash.

**Amendment 2, written before niss-superpage read (it was three loads away).** The rtl lane's R-68 words and its
correction to the PREREG's VOID rule, both received during the run:
- The word check applies to the DRAM arms only: 1 `0x80240000`, 2 `0x802c0000`, 3 `0x80340000`, 4 `0x803c0000`, 5-8 the
  same four (4 KiB aliases), 9 `0x80240100`, 10 `0x80007000`, 11 `0x80240180`, 13 `0x80240200`, 14 `0x802c0200`, 15
  `0x80240200`. These are from the simulation log `b1-wt/.sup/sc-b1-ni/niss-superpage.log`; arm 1's load from VA
  0xc0240000 returning 0x80240000 was read there.
- Arms 12 (bootrom), 16 and 17 (CLINT mtimecmp) read device contents. They are graded by cycles only, and their words
  are reported as read. Arm 17's word must equal arm 16's on the same boot; a mismatch is a finding.
- The witness is position 1, the trap count position 36, and it must be 0. The ratio keys are unchanged.
- The words are in `<mnt-hdd>/capstone/silchk-b2/r68-paddrs.json`, and the grader is `grade-silchk.py` beside it. The
  grader was negative-tested on fabricated fixed, pre-fix and void-control runs before any reading.

## Reading: the control passes, and R-60, R-72, R-32 and R-68 read their fixed vectors on silicon (board 2, resident P1, 2026-11-21 02:43-02:49 local)

Run: `<mnt-hdd>/capstone/silchk-b2/run-20261120T1843/`. It holds `chain.log`, each image's `run.log` and `uart.txt`, and
`grade.txt` from `grade-silchk.py ... r68-paddrs.json`.
- Every image's run log reads back the resident bitstream as P1's
  `caplifive_…-pagedtbl_d83d3a633.bit`. The console exposes no sha256, so the match is by name.
- The run used booking 1391 and unbooked it. The marker was touched at 02:49:31.
- Each load took about 52 s, and each saw `SUPTEST END`.

| Image (sha256 prefix) | Reading | Grade |
|---|---|---|
| va-runv f0ed3fac351b60f3 | SR 15; its 15 values exactly | control PASS |
| alias-store 46ca6a4475b06ec0 | `0x5a5a 0x1234 0x0 0x1 0x18 0x5a5a 0x7 0x58` | **R-60 CONFIRMED FIXED ON SILICON** |
| pcc-end-in 6e67467e0074c440 | `0x123 0x1c 0x4 0x4 0x1` | control as predicted |
| pcc-end-out 1dc587cd8489fe89 | `0x0 0x1c 0x0 0x0 0x1` | control as predicted |
| pcc-end f0935c425401b34d | `0x0 0x1c 0x0 0x0 0x1` | **R-72 CONFIRMED FIXED** |
| shrinkto 34311f6d26259cb1 | the 16 fixed values (position 6 `0x40`, not the pre-fix `0x3f`) | **R-32 CONFIRMED FIXED** |
| niss-superpage f34bd6e59092b95c | SR 36, witness `0xa00000800`, traps 0 | **R-68 FIXED by the ratio keys** |

**R-68 detail.**
- Ratios: arm1/arm2 250/250 = 1.00, arm10/arm9 275/275 = 1.00, arm17/arm16 1817/1818 = 1.00.
- All 14 DRAM arms read the simulated paddr words.
- The CLINT words of arms 16 and 17 are equal (0x0).
- Arm 12 reads the board's bootrom doubleword 0x3049107300800913. That differs from the simulation's 0x1f4141300100413,
  as the rtl lane said a device read may, and it is not graded.
- Cycles per arm 1-17: 250 250 250 250 272 272 272 272 275 275 250 1832 304 326 328 1818 1817.
- Its standing residual (ISSUES R-68) is unchanged by any reading: the test cannot tell the fix from deleting the splice.

**For bitstream 1:** these are readings on P1. They close the four entries for bitstream 1 by the byte-identity of the
four fixes' files plus the ccsrrw source check above, not by a reading on bitstream 1.
