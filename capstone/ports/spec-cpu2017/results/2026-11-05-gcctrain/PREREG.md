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

## Addendum 2 (5 Nov 17:15, before board 2's run; board 1's Plain boot started 17:06)

**The Temporal QEMU gate failed, and the cause is the node table size.**
- The Temporal run stopped with `cpugcc_r_base.capstone-lifetimes: out of memory allocating 4064 bytes` at 9.8 G
  instructions, rc 1. The launcher read `node_capacity=65536 node_growths=0`. The run's directory is kept as
  perf/gt/q-T.16bit-oom.
- The all2 and evaluation modules are built with the default 16-bit index, a 65,536-record flat table. That fits
  every test input. train01 needs more: census-fix2-3g ran it on an 18-bit module (262,144 records, 4 MiB table).
- Bitstream 3 and C.5a carry the 18-bit table. A board train run already used it: mcf train on bitstream 3 read
  `node_bytes=4194304 node_capacity=262144`.
- A capacity stop, not a fault: the malloc failure is the ENOMEM of the flat table's ceiling.
- No board call was made. The gate wrapper did not start board 2.

**Fix.**
- Module 65ede944a87ea742: the same source (3fbed8df1fb8) built with `-DCAPSTONE_VM_BENCH_HPM
  -DCAPSTONE_VM_INDEX_BITS=18`, 0 two-byte instructions.
- Its QEMU build 1c29ae65c8d3c57b is in kit-etq3h18, and the Temporal gate reruns on it (q-T).
- bkit26-gt now carries the 18-bit module, and gt-run.sh checks its hash.

**The Plain arm was not affected.**
- Board 1's Plain boot baked bkit26-gt at 17:06, before this change, with module b5ea7d846154fa47 (16-bit). Its
  hash checks passed against that module.
- Plain never opens a context, so the table width does not touch it. It still loads the module for
  `hpm_snapshot`, which is identical in both builds.
- The two arms therefore ran different module builds, differing only in the index width.

## Addendum 3 (5 Nov 17:42)

The rerun Temporal gate passed at 17:31: rc 0, specdiff 1/1 PASS, `node_capacity=262144`, peak 48,746 pages.

**Board 2's first boot made no measurement (17:34).**
- The boot baked, then stopped at the driver: "FPGA_BOARD=2: the console URL given is not board 2's". No program ran
  (`NO SPEC STEP`), and board 2 was flashed back to bitstream 3. The output is kept as out-gt-b2.nourl-1734.
- The cause: the kit's board-run.sh reads `BOARD_URL_FILE`, which defaults to board 1's file. Board 2 kits set it in
  board.env, and bkit26-gt, derived from a board 1 kit, does not.
- Both drivers (gt-run.sh, ns-run.sh) now export `BOARD_URL_FILE` with the board's own file. boardvm runs board-run.sh
  with the driver's environment, and sourcing board.env does not unset it.
- Board 2 is relaunched at once.
