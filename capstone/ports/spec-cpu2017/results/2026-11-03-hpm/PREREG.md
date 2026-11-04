# CVA6 counters inside SPEC contexts: first board boot, pre-registration

This file is written and pushed before the boot.

**What changes.** This boot makes one change to the 3 Nov bitstream-3 SPEC test boot of imagick: the kit. The rest is
the same: the stage `spec/stages/538.imagick_r-test` (tree-board images, MapLife and Full in one stage), the control
crc.K0, and bitstream `caplifive_vp2c_e7af48457`.
- **Kit.** `t3-smoke/bkit16-hpm-b3-b1` = `bkit12-perf-b3-b1` with two parts replaced.
  - **Module.** `capstone_vm.ko` is `2f46170a118fda1d`: runtime/bench-hpm 6f7356368bb1, `-DCAPSTONE_VM_BENCH_COUNTERS
    -DCAPSTONE_VM_BENCH_HPM`. Built without the HPM define, it gives the kit module's `.text` (6953e0dbd4f5, kit
    9c60fb53).
    - It reads hpmcounter3..8 around every CSRUNV, outside the cycle/instret reads.
    - It prints `capstone_vm: hpm hpm3=.. hpm8=..` per context, after the existing bench line, which is unchanged.
  - **Firmware.** Wrapper caplifive-opensbi hpm-events 8a39f82, with `-DCAPSTONE_HPM_EVENTS` added to the kit's defines.
    - `mstatus_init` sets mhpmevent3..8 to 1, 2, 4, 10, 22, 14 and clears mcountinhibit[8:3].
    - Without the define the firmware is byte-identical to the baseline (1ac724e63196ff21).
- **Events.** The numbers come from capstone-ariane perf_counters.sv at e7af48457, and they were confirmed on silicon
  tonight by the hpm bare kit, which read the exact fields on board 1 at 19:25-19:35:
  - hpm3: L1I miss (event 1)
  - hpm4: L1D miss (event 2)
  - hpm5: DTLB miss (event 4)
  - hpm6: branch mispredict (event 10)
  - hpm7: issue stall (event 22)
  - hpm8: scoreboard full (event 14)
- **Runner.** `perf/hpm/sbhpm/hpm-boot.sh` books board 1 at the SPEC chain's next stage end and runs
  `sbhpm/spec-board.sh`. That is the lane's spec-board.sh with three changes:
  - the bkit16 kit;
  - the module hash and the HPM define are checked;
  - a pre-taken booking is passed in.
  - The stubbed dry run reached the runner with the expected arguments and released the booking.

**QEMU gate (done).** lbm test ran in both arms on spec/kit-hpmq-{base,hpm} (QEMU-kernel builds of the same source):
- rc 0, specdiff PASS, outputs identical in both arms;
- the bench line is fresh;
- lifetimes instret is equal to base exactly (7,278,939,488);
- the hpm line reads 0 on QEMU, which programs no events.

The first gate run printed the new line as a second `capstone_vm: bench` line. That broke the stage's freshness check,
and it is why the line is now `capstone_vm: hpm`.

**Predictions** (baseline: board run out/538.imagick_r-test, bitstream 3, 3 Nov):

| # | Prediction | Baseline value |
|---|---|---|
| P1 | The control crc.K0 passes. Both arms are rc 0, specdiff PASS, and their outputs equal the twin's (census-q34). | |
| P2 | run0 instret is within 0.1 % of the baseline: counting changes no instruction inside the window. | MapLife 108,724,600; Full 108,744,930 |
| P3 | run0 cycles are within 2 % of the baseline. This is one run; a larger move would mean the extra reads or the counters cost cycles. | MapLife 402,555,914; Full 395,108,674 |
| P4 | Every context prints one `capstone_vm: hpm` line. In run0 of both arms, hpm3, hpm4, hpm5, hpm6 and hpm7 are > 0 and each is < cycles; hpm7 < cycles. hpm8 is reported, with no prediction (it may be 0). | |
| P5 | The stage reads the bench line fresh in every command (no "NOT fresh"). | |

**What each outcome means.**
- **All six counters read 0 (P4 fails).** The firmware did not set the events. Check the firmware build's defines and
  the wrapper commit in fw.log. Nothing is concluded about the programs.
- **Oops, illegal instruction or no boot.** The boot is VOID. Board 1 goes back to bkit12, and the defect goes to a repro
  folder.
- **P2 or P3 fails.** The instrument perturbs the window. Counter readings are then reported only with that caveat, and
  the counter build does not replace the timing build.
- **Everything holds.** The counter kit runs the evaluation set (lbm, nab, xz, mcf, imagick, gcc test), MapLife/Full on
  board 1, then the ObjBounds kit bkit16-hpm-objb-b2 on board 2 after its own first boot.

**Limits.**
- N = 1.
- Counters have no privilege filter, so a window also counts monitor work inside the CSRUNV, as cycle does.
- hpm5 (DTLB) counts the S-mode translation the context runs under.

## Addendum 1 (3 Nov 23:50, after the first boot held P1-P5; result in RESULT.md)

Two more boots go on the same kit, bkit16-hpm-b3-b1. Each is booked in the next board 1 chain gap by
`sbhpm/hpm-boot2.sh` (grab-gap.py) and runs one after the other.
- **gcc test, Full.** Stage `spec/stages-board/502.gcc_r-test-capstone-lifetimes`, the 3 Nov board stage. Twin
  census-q34. Baseline (out/502.gcc_r-test-capstone-lifetimes): cycles 205,767,280, instret 30,690,680.
- **imagick test, repeat.** The same stage as the first boot. It tells whether the arms' L1I-miss and mispredict
  difference (716 k vs 489 k; 2.01 M vs 1.61 M) is reproducible.

Predictions:
- P1-P5 as above, against these baselines. For gcc, P4 holds for its one arm.
- **P6 (repeat).** If the difference comes from the arm, the repeat again reads MapLife L1I > Full L1I and MapLife
  mispredicts > Full mispredicts. Each arm's L1D stays within 1 % of the first boot. If either ordering flips, the
  difference is run-to-run variation, and per-arm counter differences of that size are not read as arm effects.

## Addendum 2 (4 Nov, before the boots): imagick dev and devmk with counters

**The question.** The compiler evaluation's step 2 (+movc-keeps-integer-source) cuts imagick's cycles 12 % for 3 % of its instructions. Silicon saves 23-36 cycles per removed STC/LDC stack-copy pair. The sim reads about 8 cycles per pair, and that figure does not move with memory delay (the rtl lane, delay 12 against 40). Which counter carries the gap?

**The boots.** Two boots on kit bkit16-hpm-b3-b1, bitstream 3, board 1, run by c5a-b1.sh before the C.5a flash through `sbhpm/chain-board.sh` into `spec/board/out-hpm-eval`:
- `stages-eval-b1/538.imagick_r-test-dev`;
- `stages-eval-b1/538.imagick_r-test-devmk`.

Both are both-arm stages, with the twin census-q34.

**Baselines** (out-hfl, run command, cycles / instret):

| | MapLife | Full |
|---|---|---|
| dev | 350,906,604 / 100,882,666 | 353,505,467 / 100,905,319 |
| devmk | 308,778,978 / 97,817,914 | 310,135,911 / 97,839,264 |

**Predictions.**
- P1-P5 as in the first counter boot, against these baselines: instret within 0.1 % and cycles within 2 %.
- **H (reported, with one bound).** Between dev and devmk in each arm, the drop in issue-stall cycles (hpm7) is at least half the drop in cycles. If it is smaller, the saving is not in issue stalls, and the other counters' drops are compared by size.

## Addendum 3 (4 Nov 16:56, before the boots): where Addendum 2's boots run

The C.5a session moved to board 2, and its board 2 runner has no bitstream-3 counter kit. Addendum 2's two boots therefore run on board 1 after its fault-ahead run ends, with `perf/fa/hpm-after-fa.sh`. That script waits for fa-b1.sh, requires its DONE line, then runs `sbhpm/chain-board.sh 1` over the imagick dev and devmk stages.

**What makes them matter more now.** On board 1, bitstream 3, the slot-bare kit graded AS EXPECTED. An isolated STC→LDC pair costs 16.97 cycles on silicon (sim 17.38), yet removing one pair from compiled code saves 23-36 cycles. The excess lies in the code around the pair, and these counters are the instrument for it.

Predictions are unchanged.
