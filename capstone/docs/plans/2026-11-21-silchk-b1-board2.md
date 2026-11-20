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
