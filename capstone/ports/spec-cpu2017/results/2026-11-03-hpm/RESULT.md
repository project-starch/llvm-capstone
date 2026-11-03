# CVA6 counters inside SPEC contexts: first board boot, result

The boot was pre-registered in PREREG.md (c0623c713fef).
- **Run.** Board 1, bitstream `caplifive_vp2c_e7af48457`, kit `bkit16-hpm-b3-b1`, module `2f46170a118fda1d`. Booking 631,
  23:31:00-23:42:24 on 3 Nov.
- **Stage.** `spec/stages/538.imagick_r-test`, tree-board images. MapLife is the `capstone-spatial` arm, Full the
  `capstone-lifetimes` arm.
- **Verdict.** Every pre-registered reading holds.

## Booking

The first runner (`sbhpm/hpm-boot.sh`) booked when an "SB END" line appeared in the chain log, and it missed the 22:14
gap with rc 3. The previous stage's DRIVER rc line, its SB END and the chain's next BEGIN all fell at 22:14:33, so the
chain had rebooked before the runner tried.

The boot that ran was taken by `perf/hpm/grab-gap.py` (outside the repo):
- it waits for the chain stage's `guest/SEQ` "DRIVER rc=" line, which board-run.sh writes after its one driver exits;
- it takes the board's host flock, which board-busy.py reads as busy, so the chain waits;
- it books (2 tries, 0.4 s) and releases the flock.

It never takes the flock while a booked stage's driver may still connect. Its stubbed dry run covered every branch.

## Predictions

| # | Reading | Verdict |
|---|---|---|
| P1 | Control crc.K0 PASSED. Both arms 2/2 commands, rc 0, specdiff 1/1 PASS. Outputs identical across the arms and to the twin. | holds |
| P2 | run0 instret: MapLife 108,724,988 (baseline 108,724,600, +0.0004 %); Full 108,745,690 (108,744,930, +0.0007 %) | holds |
| P3 | run0 cycles: MapLife 402,422,394 (402,555,914, −0.03 %); Full 395,899,972 (395,108,674, +0.20 %) | holds |
| P4 | 5 contexts, 5 `capstone_vm: hpm` lines. hpm3-7 > 0 and < cycles in run0 of both arms; hpm8 2.6 M | holds |
| P5 | bench_line_fresh true in all 4 commands; no "NOT fresh" | holds |

## Readings (in-context, per command)

| Context | Cycles | Instret | L1I miss (hpm3) | L1D miss (hpm4) | DTLB miss (hpm5) | Mispredict (hpm6) | Issue stall (hpm7) | Scoreboard full (hpm8) |
|---|---|---|---|---|---|---|---|---|
| control crc.K0 | 230,146,347 | 71,462,906 | 59,114 | 7,248 | 1,326 | 20,162 | 123,584,751 | 223,600 |
| MapLife run | 402,422,394 | 108,724,988 | 716,027 | 1,480,440 | 307,398 | 2,007,554 | 161,358,330 | 2,622,281 |
| MapLife val | 288,243,096 | 83,007,133 | 396,061 | 471,684 | 9,762 | 194,225 | 130,858,282 | 329,998 |
| Full run | 395,899,972 | 108,745,690 | 489,485 | 1,481,834 | 307,870 | 1,610,034 | 164,711,046 | 2,632,431 |
| Full val | 287,226,492 | 83,028,149 | 395,829 | 448,968 | 9,146 | 192,202 | 130,302,847 | 344,286 |

Rates for the run command, per 1,000 instructions, with issue stall as a share of cycles:

| Arm | CPI | L1I | L1D | DTLB | Mispredict | Issue stall |
|---|---|---|---|---|---|---|
| MapLife | 3.70 | 6.6 | 13.6 | 2.8 | 18.5 | 40.1 % |
| Full | 3.64 | 4.5 | 13.6 | 2.8 | 14.8 | 41.6 % |

## Limits

- N = 1.
- The counters have no privilege filter, so the monitor's work inside CSRUNV is counted along with the program's.
- The arms differ in L1I misses (716 k vs 489 k) and mispredicts (2.01 M vs 1.61 M) at equal instret and L1D. One boot
  cannot say whether that difference comes from the arm or is run-to-run variation. A repeat will tell.
- Issue stall is CVA6's event 22. Its silicon count matched simulation on the hpm bare kit (1530). It is not yet split
  by cause.
