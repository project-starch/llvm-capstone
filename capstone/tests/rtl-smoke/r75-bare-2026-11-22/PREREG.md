# R-75 on silicon: pre-registration (2026-11-22, written before any board run)

Bitstream 1 = `caplifive_…-pagedtbl-r76_9e801ab5e.bit` (sha256 01707b37…). R-75 (d) = `caplifive_…-pagedtbl-r76-r75d_1c5d31474.bit`
(sha256 2d2ba92c2e193dc4…). Vectors come from simulating THESE images (`out/*.elf`) through the testharness on each
tree (S12_MEM_DELAY 12, time_out 3,000,000, dirty=no; read from the recorder area by board-vector.py). In simulation
every kit image ends in CAP_PASS's spin, so the harness's "FAILED at the cycle limit" is expected for all of them.

| image | sha256 (16) | board_rec | bitstream 1 (9e801ab5e) | R-75 (d) (1c5d31474) |
|---|---|---|---|---|
| `va-runv.bin` (control) | f0ed3fac351b60f3 | 0x8000c000 | SR 15: `0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb 0x0 0x9` | the same |
| `r75-trapz.bin` (matched control) | 311e56388094b2ad | 0x80012000 | SR 3: `0x2 0x80004000 0x5` | the same |
| `r75-trap.bin` (R-75) | 73ce33d96fac8e91 | 0x80012000 | **NO SR line**: one value recorded (0x2), then the M handler re-faults at its own entry (0x80003000) for ever (45,118 entries in 3 M cycles) | SR 3: `0x2 0x80004000 0x5` |

**What the kit measures, and its limit, stated before any run.**
- The board harness's CAPPRINT uses the M-only debug mints, so it records only prints executed in M. va-fetchsplit's
  first S-mode print takes a load-access fault in the recorder itself (0x80002070), and the M handler then ends the
  run through CAP_PASS. That is why the arms record three values, not the in-tree arm's thirty.
- R-75 is armed at that trap into M. On bitstream 1 it loops there, which is R-75's signature: the handler re-faults at
  its own entry. On (d) it passes.
- r75-trapz is the same scaffold with -DCS4_ZERO. It takes the same recorder fault and returns on both trees, so it is
  the matched control that the scaffold reaches the point.
- The kit grades R-75's point, not the arm's whole vector. The trace labels the handler re-entries LD_ACCESS_FAULT; that
  cause label was the source of an earlier R-75 retraction, so the repeated handler PC is the signature, not the label.

**Board procedure** (the runtime lane's rules, 2026-11-22). One image per JTAG load, the control first in each session:
1. Bitstream 1 resident (read back): va-runv, then r75-trapz, then r75-trap LAST. Each load's timeout is at most 150 s.
2. va-runv again, to re-confirm the board after the expected wedge.
3. Flash R-75 (d); read the resident name and its sha256 back.
4. va-runv, r75-trapz, r75-trap, then r75-trap again in a separate load: two returns against the expected non-return on
   bitstream 1 make a matched pair for a one-cycle race.
5. The runtime lane's short perf null (makerand) on (d).

**Grading.**
- VOID unless every va-runv load reads SR 15 and its 15 values.
- r75-trapz must read its SR 3 vector on both bitstreams. Otherwise the scaffold did not reach the point, and the R-75
  reading is VOID.
- **R-75 REPRODUCED ON SILICON:** bitstream 1's r75-trap prints no SR line within its timeout, while r75-trapz in the
  same session returns.
- **R-75 FIXED ON SILICON by (d):** both r75-trap loads on (d) read the SR 3 vector.
- **R-75 NOT REPRODUCED:** bitstream 1's r75-trap returns. That is a finding (the race did not arm on silicon), NOT a
  pass, and (d)'s readings then show no regression only.
- Cite readings by image hash, never by label.
