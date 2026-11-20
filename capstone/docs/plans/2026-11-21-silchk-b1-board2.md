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
