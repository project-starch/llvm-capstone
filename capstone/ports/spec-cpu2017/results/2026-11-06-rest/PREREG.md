# The two remaining SPEC test programs, perlbench and x264, in the final configuration: pre-registration

Written and pushed before the board calls (6 Nov, 01:00). The lead's goal for today is every SPEC test under 1.20×
Plain. These two have no measurement in the final configuration yet.

## The runs (kit bkit26-gt: C.5a, module 65ede944a87ea742, 18-bit, counter firmware, all2 params, exact-image launcher)

**Board 1: perlbench test, Temporal (devmk images).**
- Stage `perf/rest/stages/500.perlbench_r-test-capstone-lifetimes-devmk`, built by make-stage.py from
  tree-boarddevmk with the test options of make-stages-hfl.sh: strip, native harness, keep-lib.
- `after-gate.sh` starts the driver only once the QEMU gate (`qgate.sh`, kit-etq3h18) has completed every command,
  rc 0. A specdiff failure there is recorded, not gating. Before the heapfix, perl's reg_email failed specdiff
  on silicon and on the twin.
- It then waits for ns-run.sh 1 (pid 1171962) to end with DONE.

**Board 2: x264 test, Plain.** `spec/stages-test-plain/525.x264_r-test-plain-rv64`, the 3 Nov Plain stage. It runs
after perf/mx's Plain mcf (driver pid 1370451) ends with DONE.

**Still owed:**
- x264 Temporal goes on the next free board.
- perlbench Plain is the 3 Nov board reading (out-hfl: 501.52 + 22,809.06 s, specdiff 2/2).

**Preempted for these runs.** Two drivers were killed while still waiting, before any board call. Their changes go
into the next gcc and mcf runs:
- perf/mx board 1, mcf with resume_quantum;
- perf/gx board 2, gcc train with the quiet launcher.

## Readings and predictions

| # | Prediction | Point estimate |
|---|---|---|
| R0 | each arm: rc 0, control passes; QNT0 007A1200 | |
| R1 | perlbench Temporal run below the 30 Oct baseline (939.67 + 43,358.42 = 44,298 s) | 28,000-36,000 s: 1.2-1.55× Plain's 23,311 s. Run 1 had 1.37 M exits on 30 Oct, so the outside levers carry it |
| R2 | x264 Plain run within 30,000-48,000 s | Capstone's 30 Oct run held 43,160 s of in-context cycles for 543.7 G instructions |

**Reported:**
- run time against Plain and against the 30 Oct baseline (x264 Temporal 51,855.95 s);
- inside and outside time, and exits by kind.

x264's board stage has no imagevalidate step (by design, since 30 Oct), so specdiff reads 0/3. Its outputs are compared with the twin's.

**Dry run** (perf/rest/dryrun, board calls stubbed): A-gateok, with a gate log reading rc 0, starts the driver; the
chain stub checks every list path. `board-env-check.py` passes for both boards' lists.

**Limits.** N = 1. Plain and Temporal of x264 run on different boards and days.

## Addendum 1 (6 Nov 01:35, before either board call)

**The images change to the arena-cache runtime, and the kit to bkit27-gtq.**
- Both programs are relinked with the arena-cache SDK (results/2026-11-06-arena): perlbench bd186c2bd219b194 (stripped
  as make-stage does), x264 00565bd74f342fc4.
- A relink of the same recompiled objects with the production SDK reproduces the tree's exes byte for byte: perl
  1c3a6f43, x264 1c57ad01.
- bkit27-gtq is bkit26-gt with the quiet launcher.
- **Reason:** gcc's exits were allocator map/unmap, and perlbench's 1.37 M exits on 30 Oct are likely the same.
  Measuring the old images first would cost a second 10-hour run.

**The QEMU gates are now `q-perl-ac` and `q-x264-ac`** (kit-profq). The old-image gates keep running as twins.

**The order changes:**
- Board 1, after perf/mx's mcf (driver pid 1521568): perlbench-ac, then x264-ac Temporal, as one list.
- Board 2, after perf/ga's gcc runs: x264 Plain.

**R1 is unchanged.** Its point estimate is now open: the cache's effect on perl has not been measured, even on QEMU.

## Addendum 2 (6 Nov 15:15, before either board call): both programs in gcc train's best configuration

The lead asked for the two remaining tests to run in the best configuration. That is now gcc train's queued one
(results/2026-11-06-arena Addenda 3-7):
- **compiler:** llvm-c3 (llvm-c2 + llvm-capstone perf/shrinkto 1836a2142f7d), `-capstone-shrinkto-globals
  -capstone-shrinkto-end-adjust=1` (C.5a only, R-32);
- **runtime:** SDK ac12 (arena cache, slot cache 65,536, simple 16-bit hash, null-capability memset);
- **module parameters, per stage:** step budget 10 s, resume_quantum 1, retire_exit_sweep 2, arena_pin 1 (ga2's
  block, each read back and restored).

Workload sources are unchanged. The objects are recompiled from each program's production compile list with only
the compiler swapped (perf/best/build-best-c3s1.sh).

| program | image (stripped) | QEMU instructions (kit-profq) | outputs |
|---|---|---|---|
| perlbench | 95f4bbb17e74c543 | run1 96.13 G, against 109.51 G for -ac (−12.2 %); c2ac12 alone 100.45 G | test.out, makerand.out = -ac's byte for byte; specdiff 1/2 as every arm (japh/abigail.t) |
| x264 | d003385b71056a8a | gate running at the time of writing | |

**Order:**
- **Board 1:** perlbench-ac continues. x264-ac is replaced in `perf/rest/b1.list` by
  `perf/best/stages/525.x264_r-test-capstone-lifetimes-devmk-c3s1`. The edit is in place: same inode, line 1
  unchanged, so the running chain reads it. If x264's QEMU gate fails before the boot, the line goes back to -ac, and
  that is recorded here.
- **Board 2** (`perf/best/b2-seq.sh`, one task): gcc train (ga2) → the rtl lane's R-32 SHRINKTO readback bare boot
  (land/rtl-r32-kit 624924575c53, graded by its own PREREG) → the THP mcf runs (results/2026-11-06-thp, unchanged) →
  perlbench in the best configuration (`perf/best/b2.list`). The waiting THP driver (pid 2024109) is killed before
  any board call and runs again inside the sequence, unchanged.

**Predictions:**
- **B1 (perlbench best):** run ≤ 0.92 × perlbench-ac's board run. Point −10…−15 %: inside −12 % of instructions, plus
  arena_pin and the 10 s step budget outside. perl's 318 contexts make the exit-sweep and launch costs uncertain.
- **B2 (x264 best):** run ≤ x264's 30 Oct Temporal 51,855.95 s. A point estimate follows with the QEMU gate.
- Reported: both against Plain (perl 23,311 s; x264 from out-rest-b2).

**Dry run** (perf/best/dryrun/A-ok, every board call stubbed): the sequencer waits for the pid, runs R32, THP and
BEST, and every list path passes the chain stub. board-env-check passes for the perl stage.
