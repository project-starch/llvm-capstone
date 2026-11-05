# gcc train (train01), Capstone Temporal against Plain, with every counter: pre-registration

Written and pushed before any board call (5 Nov, 17:10). Driver `gt-run.sh`, grader `gt-grade.py`, lists `b1.list`
(Plain) and `b2.list` (Temporal), the stages' run.sh in `stage-T/` and `stage-P/`, and the rusage wrapper in
`stage-P/ru.c`.

## Why

The lead asked for a gcc train row on slide 11, the final configuration only, with every performance metric.
- gcc test is 18 M instructions, so fixed costs dominate its 3.22×.
- Full gcc train does not fit on the board: 200.c needs 1.62 GiB and scilab.c 1.24 GiB, against about 963 MiB of
  Linux memory (census-fix2-3g). The set was cut to train01 on 3 Nov.

## The run

- **Stages** (make-stage.py, wt-spec, `--size train --commands 2`, that is train01 only):
  - `502.gcc_r-train01-capstone-lifetimes-devmk`: tree-boarddevmk, `--strip`. Its executable d591641944f6 is the
    devmk gcc of the evaluation set.
  - `502.gcc_r-train01-plain-rv64`: tree-plain, with rdctr.
- **Configuration: all2's**, on C.5a. That is the devmk images, fault-ahead 64, `retire_exit_sweep=3`, `image_rootfs=1`,
  the 80 ms step budget, the exact-grant launcher 0bde5fd8e0c9c374 and quantum 8,000,000.
- **Kit bkit26-gt** = bkit23-all2-b1 with two changes:
  - **Module b5ea7d846154fa47.** This is perf/exit-timing-vp2 3fbed8df1fb8 built with `-DCAPSTONE_VM_BENCH_HPM`: the
    exit-timing counters of results/2026-11-05-nosweep Addendum 1, the per-context `capstone_vm: hpm` line, and a new
    read-only `hpm_snapshot` parameter. That parameter gives cycle, instret and hpm3..8 at the moment it is read.
    `resume_quantum` stays off.
  - **The counter firmware.** Wrapper wt-wrap-hpm 8a39f82, that is wt-wrap-s26 plus one commit, with
    `-DCAPSTONE_HPM_EVENTS` added to the kit defines. It sets hpm3-8 to L1I miss, L1D miss, DTLB miss, mispredict,
    issue stall and scoreboard full (results/2026-11-03-hpm, confirmed by the bare kit).
- **What each command records.**
  - **Before and after, both arms:** the machine-wide `hpm_snapshot`; /proc/vmstat pgfault, pgmajfault, pgalloc and
    pgfree; /proc/stat ctxt, intr and processes; MemFree.
  - **Plain also:** rusage (maxrss, minflt, majflt, user and system time) through `ru`.
  - **Temporal also:** the module's in-context cycle, instret and hpm counts, the per-exit timing split, exits by kind,
    teardown, and the launcher's faults, pages and steps.
- **Boards.** Board 2 runs Temporal after stage A's session (dt-b2.sh) ends with DONE. Board 1 runs Plain after all2's
  session ends with DONE. Each board flashes C.5a before and bitstream 3 after. The no-sweep run follows on each board
  (its drivers then wait on these).

**QEMU gate** (kit-etq3h, the QEMU build ee8f0d1c6c2fbaf0 of the same source):
- Plain: rc 0 and specdiff PASS. The snapshots and rusage are present (minflt 89,124, maxrss 98.9 MB). hpm reads 0,
  since QEMU programs no events.
- Temporal: Addendum 1.

## Predictions (`gt-grade.py`)

| # | Prediction |
|---|---|
| G0 | rc 0, specdiff PASS, control passes, bench line fresh (Temporal); the four params read back; QNT0 007A1200 |
| G1 | the counters were programmed: in-context hpm3-7 in (0, cycles) for Temporal; machine-wide hpm3-7 > 0 in both arms |
| G2 | Temporal run 3,000-6,500 s; Plain run 700-2,500 s |

**Reported, with point estimates where we have them:**
- Capstone ÷ Plain run time: about 2.2-3×, against 3.22× on test.
  - Temporal: 16.2 G program instructions (QEMU twin, 3 Nov images), 227 k context entries, 52.6 k pages, so
    0.7-1.1 h inside and 0.3-0.45 h outside.
  - Plain: about 9.6 G program instructions.
- Inside against outside.
- Every counter per kilo-instruction, machine-wide in both arms and in-context for Temporal.
- Page faults, exits by kind, and time per exit.

**Limits.**
- N = 1.
- train01 only.
- Machine-wide counters include the kernel and everything else on the hart; that is what makes them comparable
  across the arms.
- hpm8 (scoreboard full) is reported only.
- There is no ITLB event in the six-counter set.

## Addendum 1 (5 Nov 17:15, before any board call of this run)

**The lead made this run the top priority, so both boards were preempted at 17:02.**
- **Board 1:** all2's mcf stage, started 16:13, about 1.5 h short of its end. The all2 session (pid 548710) was killed
  with its chain, runner and driver, and booking 716 released.
- **Board 2:** stage A's mcf stage (dt-b2.sh, pid 183797), started 14:04, about 1.5 h short. Killed the same way,
  booking 714 released.
- Neither mcf result exists; both stages go back in the queue after this run. Board 1 was left on C.5a and board 2 on
  stage A's bitstream. Each run here flashes C.5a first.

**The drivers start at once** (no waited pid).
- Board 1 starts now with Plain. Its QEMU gate is done.
- Board 2 starts with Temporal as soon as the Temporal QEMU gate (q-T, still running) reads rc 0 and specdiff PASS.
  A wrapper waits for that and runs nothing otherwise.

**A driver defect fixed before any use.**
- gt-run.sh, and ns-run.sh of results/2026-11-05-nosweep, exported board 1's console URL file for both boards.
- fpga_console.py refuses that file on board 2 ("the console URL given is not board 2's"). This board 2 driver would
  have stopped at its first board call.
- Both drivers now take `fpga-console-url-board2` on board 2.
- The stubbed dry runs could not see it, because they replace every board call.
