# R-79 PMP kit on silicon: board lane run plan (2026-11-23, before any board run)

The kit is the rtl lane's, pre-registered at dev 9d3a8afbce52 (lane land/rtl-r79-kit,
`capstone/tests/rtl-smoke/r79-pmp-bare-2026-11-23/PREREG.md`). That PREREG holds the vectors and the grading; this plan
adds only where and how the kit runs. The runtime lane granted both slots on 2026-11-23.

**Images.**
- the control: va-runv f0ed3fac351b60f3, rec 0x8000c000, from the R-75 kit at 515d5bc97c7b. Its SR 15 vector passed
  on P1 on 2026-11-21 (`2026-11-21-silchk-b1-board2.md`), and it is the PREREG'd control on bitstream 1 and R-75 (d);
- the probe: r79-pmp 8eb72553965cfc1c, rec 0x80007080, SR 17.

**Trees.** core/pmp/src/pmp_data_if.sv is blob 4a8570748e9e on every Capstone tree resident or queued on the boards.
R-75 (d)'s +33 lines in load_store_unit.sv sit behind `R75_EARLYCLR` and touch only the PC-capability tracker; read in
the diff. Board 3's riscv-base 8db8f9cda has a different pmp_data_if (d18149fb29b9) and is not used.

**Runs.** Each run books 30 min, flashes nothing, reads back the resident and STOPs before any load on a mismatch, then
loads va-runv and r79-pmp, one JTAG load each with a 150 s timeout, and always unbooks.
- **Board 2**, R-75 (d) 1c5d31474 resident. It runs right after the R-75 kit, and only if that kit logged "ALL RUNS
  DONE; R-75 (d) left resident". It runs before the runtime lane's r75n. The runtime marker r75kit2-b2.DONE is touched
  only after every booking either kit took reads BOOKED x and then UNBOOKED x.
- **Board 1**, P1 d83d3a633 resident. It runs at the boundary after the runtime lane's t201s2: the pid is polled every
  5 s, and the board must read free within 40 s. If t202fp takes the board first, the run STOPs before booking and
  falls back to board 2 after h16s, on P1.

**Grading** (`r79kit/grade-r79kit.py`, tested on fabricated pre-fix, fixed, mixed, void-control and void-kit runs):
- VOID unless the control reads its SR 15 vector;
- VOID, per the PREREG, unless r79-pmp gives SR 17 with position 0 = 0x0 and positions 12-15 = 0x77;
- CONFIRMED only on the exact pre-fix vector;
- the fixed vector means NOT REPRODUCED, a finding for the rtl lane;
- anything else is a per-position finding.
- N=2: the two boards should agree exactly, and a disagreement is itself a finding.

**Instrument checks before arming.** The runner `run-r79kit.sh` was dry-run through every branch:
- a normal run on each board;
- a readback mismatch, which unbooks with no load;
- a busy STOP;
- a bad seal;
- a missing url file;
- the pid wait, run for real, both ending and timing out.
The board 2 orchestrator ran on fake logs for all three R-75 outcomes, against a scratch marker.

## Reading (2026-11-23), board 2: R-79 CONFIRMED ON SILICON on R-75 (d)

Run `r79kit/run-b2-20261123T0130`, board 2, booking 1438 (booked 02:41:47Z, right after the R-75 kit's booking 1437
was released; unbooked 02:43:33Z). No flash. The resident read back as `…-pagedtbl-r76-r75d_1c5d31474.bit`, name only
(rc 3), before any load. `grade-r79kit.py` rc 0.

| load | image | reading |
|---|---|---|
| 1 va-runv | f0ed3fac351b60f3 | SR 15, its 15 values: control PASS |
| 2 r79-pmp | 8eb72553965cfc1c | SR 17 `0x0 0x7 0x80007008 0x77 0x0 0x0 0x1234 0x7 0x80003008 0x5a5a5a5a5a5a5a5a 0x0 0x0 0x77 0x77 0x77 0x77 0x2` |

**R-79 CONFIRMED ON SILICON.** The vector is the PREREG's pre-fix row word for word (PREREG.md:31), and positions 0 and
12-15 hold their kit checks. Read per position against the PREREG:
- a legal load with a store behind it is refused 7, with tval 0x80007008, the store's address;
- a lone sd to the W-less r_page is not refused, and the write lands (0x1234 read back);
- the mixed case faults with the following store's tval 0x80003008;
- an lr.d with nothing behind it is judged a read and allowed.

The orchestrator then touched the runtime lane's marker `r75kit2-b2.DONE` at 02:43:33Z, after both bookings (1437,
1438) read BOOKED and UNBOOKED. At 02:43Z board 1's run on P1 d83d3a633, the second of N=2, was still waiting at its boundary.
