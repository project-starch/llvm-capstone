# Board 1, one booking: CSMINT N=2 on d0afdb40f, then R-79 on the P1 line (p1-r79), then P1 restored (2026-11-24, before the run)

Pre-registration for one board-1 booking, written before the p1-r79 seal exists and before any load. The slot and the
shape are the runtime lane's. The grading keys are the rtl lane's.

## The p1-r79 build

- **Source:** capstone-ariane p1-r79 7bd984edc (origin), handed over by the rtl lane.
  - Its core change is 7214834e3 = 1c5d31474 + 09b9cf9fd, applied by anchor. It touches `load_store_unit.sv` +25/-2
    and `pmp_data_if.sv` +10.
  - Checked here: the changed lines are identical to 09b9cf9fd's, and 7bd984edc adds only `verif/` files.
- **rtl's readings on 7bd984edc (dirty=no):**
  - lint PASS, every counter equal to 1c5d31474's (LATCH 44, MULTIDRIVEN 3, UNOPTFLAT 39, loop names = reference);
  - r79-cpmp sim FIXED 21/21;
  - the r79-pmp image reads its fixed row;
  - 95-sweep 92 identical in values, cycles and RVFI.
- **Build:** full build only. The runtime lane ruled that the full build is the gate. It is built exactly as 1c5d31474
  was: no XDEF, no Sublet define, R75_EARLYCLR in-tree. It uses `synth/twc/full-chain.sh` under vivado-guard
  (40 GB, gated on mem-headroom), with a loops-only in-flight gate against 1c5d31474's own full build and the LUTLP-1
  stop.
- **Seal name:** `…-r76-r75d-r79_7bd984edc.bit`.
- **Grade:** `grade_full.py`, unchanged, against R-75 (d)'s seal 1c5d31474. Keys: placed congestion <= 5, check_timing
  loops 0, WNS in the null band, routed clean. LUT and FF are reported. rtl's pre-registration is "utilization and loops
  in 1c5d31474's class".

## The booking

Runner: `csmintkit/run-b1pair.sh` (orchestrator `orch-b1pair.sh`), run `run-b1p-20261123T2309`.

**Getting the board:**
- It waits for t202sp (nat-run3 pid 893413, /proc start time 633112588), polling every 1 s, then polls board-busy
  every 2 s.
- At FREE it starts a holder (`run_sup_bare-hold.py`, FPGA_BOARD=1), so board-busy reads BUSY until the unbook, and
  books 60 min. t202s2 waits behind it.

**Phase A, CSMINT N=2:**
- Flash d0afdb40f (sha checked), read it back, run the six loads, read it back again.
- Graded by `grade-csmintkit2.py` against the amended PREREG (`land/rtl-csmkit-amend` d919aa1316e3, on dev in
  5d45de462acf), the same keys as board 2's reading:
  - control sublet-life: SR 35, positions 0-23 and 34;
  - csmint, csmint-vm, csmint-vm-nc and csmint-pmp exact;
  - r79-pmp's fixed row.
- Prediction: exactly board 2's reading (the simulation is cycle-deterministic). Positions 24-33 are recorded, not
  graded, and are expected to differ from board 2's.
- If the readback after the loads does not name the seal, the loads are VOID.

**Phase B, R-79 on p1-r79:**
- It runs only if `prep-p1r79.sh` passes at the moment the board frees:
  - the build graded FULL BUILD PASS and its verdict is SEALED with the 7bd984edc name;
  - the kit worktrees are clean at their pins;
  - r79-cpmp is included only through a `cpmp.conf` written after reading rtl's pushed PREREG.
- Flash the p1-r79 seal (sha checked) and read it back by name. Then run the images in this order:
  1. va-runv (control);
  2. r79-pmp;
  3. r79-cpmp.
- Read it back again after the loads. Any phase-B failure skips to the restore.
- **va-runv** (R-75 kit control; wt-r75kit 515d5bc97c7b, f0ed3fac351b60f3, rec 0x8000c000):
  - On 1c5d31474 it reads SR 15 `0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb 0x0 0x9`.
  - Its p1-r79 vector is graded from rtl's r79-cpmp PREREG once that is pushed.
  - If phase B runs before that PREREG exists, the control is compared with the 1c5d31474 vector above. That is rtl's
    expectation ("unchanged"), not a sim reading, and the record will say so.
- **r79-pmp** (wt-r79kit 9d3a8afbce52, 8eb72553965cfc1c, rec 0x80007080, SR 17):
  - Graded against 9d3a8afbce52's fixed row: `0x0 0x0 0x0 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x5a5a5a5a5a5a5a5a 0x7
    0x80002008 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x77 0x77 0x77 0x77 0x3`.
  - That row on p1-r79 is a sim fact rtl sent the board lane (image sim on 7bd984edc, dirty=no, retired 1,479,916,
    count 17). It is to be committed in the r79-cpmp PREREG.
  - Reading this row on p1-r79 = R-79 FIXED ON SILICON on the P1 line (PMP side).
  - Reading the pre-fix row (first differing at position 1: 0x7) = the fix NOT effective on this seal.
- **r79-cpmp** (image sha c1efe4e23976c6cd…, rec 0x80009000, SR 21):
  - Run only on the p1-r79 seal read back by name. CAPENTER is illegal on a Sublet resident, so a run there means
    nothing.
  - Graded only by rtl's PREREG on `land/rtl-r79-cpmp-kit`, which must be on origin before the booking. Otherwise it is
    left for the next gap.

**Ordering rule:** the first failure in a phase ends that phase's results. Later loads in it are collateral.

**Restore:** flash P1 `…_d83d3a633.bit` (sha from the runtime lane's SEAL file) and read it back. The unbook happens
ONLY after P1 reads back. Otherwise HOLD: the booking and the holder are kept and the runtime lane is told.

The board-stubbed dry runs passed every branch: phase B ok, prep failing, bad seal sha, bad rec, B flash fail, B
readback fail, B post-loads readback fail, restore fail (HOLD), seal flash fail (HOLD), and RK_WAITPID without its
start time (refused).
