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

## Addendum 1 boots (4 Nov 00:59-01:21, bookings 633 and 634, grab-gap)

Both boots pass P1 and P5: the control PASSED, specdiff PASS, outputs identical to the twin, every bench line fresh.

| Boot | Arm | Cycles (baseline) | Instret (baseline) | P2 / P3 |
|---|---|---|---|---|
| gcc test | Full | 206,862,339 (205,767,280, +0.53 %) | 30,692,200 (30,690,680, +0.005 %) | hold |
| imagick repeat | MapLife | 402,930,143 (402,555,914, +0.09 %) | 108,726,992 (108,724,600, +0.002 %) | hold |
| imagick repeat | Full | 395,854,200 (395,108,674, +0.19 %) | 108,745,845 (108,744,930, +0.001 %) | hold |

| Context | L1I miss | L1D miss | DTLB miss | Mispredict | Issue stall | Scoreboard full |
|---|---|---|---|---|---|---|
| gcc Full run | 1,852,170 | 544,398 | 43,381 | 1,007,028 | 70,236,565 | 11,257,917 |
| imagick MapLife run | 720,328 | 1,485,483 | 307,855 | 2,008,567 | 161,673,061 | 2,621,815 |
| imagick Full run | 487,261 | 1,477,549 | 307,779 | 1,611,354 | 164,778,976 | 2,635,519 |

- **P4 holds.** hpm3-7 are > 0 and < cycles in every run command.
- **P6 holds.** MapLife reads more L1I misses than Full in both boots (720 k vs 487 k; first boot 716 k vs 489 k), and
  more mispredicts (2.01 M vs 1.61 M, as before). L1D is within 0.4 % of the first boot in both arms. The arms'
  L1I-miss and mispredict difference therefore reproduces. It is read as coming from the arm, which runs different
  executables, not as noise. The cause is not identified.
- **gcc Full run.** CPI 6.74. Per 1,000 instructions: L1I 60.3, L1D 17.7, DTLB 1.4, mispredicts 32.8. Issue stall is
  34 % of cycles and scoreboard full 5.4 %. gcc's L1I miss rate is 9× imagick's and 3.4× its own L1D rate. Of the
  readings so far, it is the one that bears most on direction 3 (code size).

## Addendum 2 boots (5 Nov 02:51-03:15, board 1, bitstream 3): imagick dev against devmk

**The boots.** Run command, in-context, kit bkit16-hpm-b3-b1, out-hpm-eval.
- **P1-P5 hold.** rc 0, specdiff PASS and outputs equal the twin in both arms of both stages.
- **Against out-hfl.** instret is within 0.02 % and cycles within 0.3 % (dev MapLife 351,581,908 against 350,906,604; devmk 309,615,164 against 308,778,978).

| Arm | Cycle drop dev → devmk | Issue stall (hpm7) drop | Scoreboard full (hpm8) | L1I miss (hpm3) | L1D miss (hpm4) | Mispredict (hpm6) |
|---|---|---|---|---|---|---|
| MapLife | 41,966,744 | 29,253,246 (69.7 %) | 3,148,069 (7.5 %) | 15,554 | 28,240 | −70,268 |
| Full | 43,331,324 | 29,334,919 (67.7 %) | 2,940,296 (6.8 %) | 51,766 | 31,441 | 86,180 |

**H holds.** The issue-stall drop is 68-70 % of the cycle drop, above the pre-registered half. The cache counters barely move:
- at the silicon cost of 22.1 cycles per L1I miss (hpm-8, below), the L1I drop is 0.3-1.1 M cycles of a 42-43 M drop;
- the L1D drop is about 0.1 %.

The movc flag's saving on imagick (about 1.5 M fewer stack-copy pairs, about 27 cycles each on bitstream 3) is therefore core time, mostly issue stalls. It is not instruction fetch. The code-size / L1I hypothesis for the 23-36 cycles per pair is **refuted** on imagick.

## hpm-8, the L1I counter's positive control (5 Nov 03:16, board 1, bitstream 3; Addendum 4)

AS EXPECTED, graded by kit-c950068d's hpm8-grade.py:
- exact fields OK;
- L1I misses: pass 1 515 (band 480-560), pass 2 1 (at most 8);
- cycles: pass 1 13,446, pass 2 2,075, so **22.1 cycles per L1I miss on silicon** (sim 46.8).

The L1I counter (event 1) therefore counts on silicon, and the readings above can be used. gcc's 60.3 L1I misses per 1,000 instructions at 22.1 cycles are about 1.3 of its CPI of 6.74, about 20 % of its in-context cycles, not the 40 % estimated with the sim's cost.
