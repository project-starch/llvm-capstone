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

## Before the run: two changes, both made before any reading

- **Collateral rule (the rtl lane, 2026-11-24).** Phase B's ordering rule above is narrowed: only a load that FAILED TO
  RETURN (no SR line, or no uart.txt) makes the loads after it collateral. A load that returns a wrong value is a reading,
  so r79-cpmp after a returned NOT FIXED r79-pmp is graded and flagged. This is sound because the runner gives every image
  a fresh start: `run_sup_bare.py:75-83` power-cycles, runs `monitor reset halt` and a fresh `load_image` per load.
  `grade-p1r79.py` encodes it.
- **d0afdb40f stays a bare-kit seal, never the default** (the board, rtl and runtime lanes agreed). CAPSTONE_SUBLET_FMT
  makes CAPENTER illegal (`decoder.sv`), and the monitor CAPENTERs at init (opensbi 49b0f932 `sbi_capstone_init.S:34`).
  The default waits for the integer-M monitor redesign.
- The queue behind this booking changed after it was written: nested p1n came next, not t202s2. The booking's shape did
  not change.

## The p1-r79 build: FULL BUILD PASS (4/4 keys)

Built under vivado-guard: exit=0, 1 h 33 m, peak RSS 29.71 GB. The loops-only in-flight gate passed and LUTLP-1 read 0.

- **Seal:** `…-r76-r75d-r79_7bd984edc.bit`, 11,443,722 B, sha256
  `ca3bd00e0fd4a067e90c60e24054759093674e337df16b5c64bdfb7888f4e04b`.
- **Source:** PROVENANCE names HEAD 7bd984edc01d. The working tree differs only in the four host env scripts, the same four
  as 1c5d31474's reference build.
- **Keys** (`grade_full.py` against 1c5d31474's full build):
  - placed congestion level 4 (ref 5);
  - check_timing loops 0;
  - routed WNS -12.619 ns (ref -13.598, Δ +0.979, inside the ±3.74 band);
  - routed clean, 242,229/242,229.
- **Reported, not graded:**
  - LUT 186,920 (91.72 %, Δ -992);
  - FF 76,852 (Δ +59);
  - slices 50,645 (99.40 %, Δ +25);
  - TNS -195,033 (ref -224,506);
  - WHS +0.049.

## Result: booking 1463 (board 1, 2026-11-24 01:09:57-01:28:50Z), run `run-b1p-20261123T2309`

### Phase B: all three loads EXACT, i.e. R-79 FIXED ON SILICON on the P1 line, PMP and CPMP sides, N=1 (the board lane's grade, and the rtl lane's independent grade from the raw uart.txt, landed on dev in 16c131fdb5f6)

- The seal read back by name as `…-r75d-r79_7bd984edc.bit` after the flash and again after the loads.
- `grade-p1r79.py` (PREREG 172d21085037) graded all three loads EXACT, rc 0.
- Each vector was read again by hand from the raw uart.txt. Each load has one SUPTEST BEGIN, and its image sha16 in
  chain.log equals the table's.

| load | image sha16 | SR | read |
|---|---|---|---|
| va-runv (control) | f0ed3fac351b60f3 | 15 | `0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb 0x0 0x9` (EXACT) |
| r79-pmp | 8eb72553965cfc1c | 17 | the 9d3a8afbce52 fixed row (EXACT) |
| r79-cpmp | c1efe4e23976c6cd | 21 | `0x0 0x0 0x0 0x5a5a5a5a5a5a5a5a 0x7 0x80003000 0x5a5a5a5a5a5a5a5a 0x7 0x80003000 0x5 0x80004000 0x77 0x0 0x0 0x0 0x7 0x80003008 0x4321 0x99 0x5a5a5a5a5a5a5a5a 0x4` (EXACT) |

The rtl lane grades this run independently from the raw uart.txt before anything lands on dev.

### Phase A: CSMINT N=2 on d0afdb40f is VOID (infrastructure, not a reading)

- The seal read back by name after the flash and again after the loads.
- **The control sublet-life never ran.** run_sup_bare.py reached `reset halt` and `load_image`, and then `set $pc` timed
  out: `[riscv.cpu] Timed out after 120s waiting for busy to go low (abstractcs=0x8001102)`. The load left no uart.txt
  and exited rc=1.
- This was the first load after the flash and its power cycle. It is the JTAG debug-module failure recorded in
  `SILICON-BLOCKER.md:4062` and history 19-07-2026, and it says nothing about the image or the design.
- The PREREG voids the kit without its control.
- The other five loads each had a fresh power cycle and load. All five read EXACT, and their SR/SV streams are
  byte-identical to board 2's run (booking 1457) for csmint, csmint-vm, csmint-vm-nc, csmint-pmp and r79-pmp. They are
  recorded here and not graded.
- CSMINT stays CONFIRMED ON SILICON at N=1, on board 2.
- No retry: the runtime lane declined to reorder the queue for it. d0afdb40f cannot become the default at any N, so a
  second reading is for the record only. It is to be offered a natural gap if one opens.

### Restore

- P1 `…_d83d3a633.bit` was flashed (sha checked) and read back by name: `RESTORED … rc=3`.
- Then UNBOOK 1463 returned 200, and the holder stopped. No HOLD.
