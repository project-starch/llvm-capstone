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
