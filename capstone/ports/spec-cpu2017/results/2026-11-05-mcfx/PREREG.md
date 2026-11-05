# mcf test: the outside-lever bundle, and Plain with every counter: pre-registration

Written and pushed before either board call (5 Nov, 23:10). Driver `perf/mx/gt-run.sh` (a frozen copy of perf/gtm/gt-run.sh,
run with `GT_O=$CAPSTONE_WORK/perf/mx` and `GT_OUT=$CAPSTONE_WORK/spec/board/out-mx-b<board>`), grader `mx-grade.py`,
stage scripts `stage-x/run.sh` and `stage-P/run.sh` (written by `mk-stages.py`), lists `b1.list` and `b2.list`.

## Why

mcf is the program furthest from Plain after all optimisations: 11,573.79 s against Plain 9,082.86 s (results of
5 Nov, out-gtm-b1: kit bkit26-gt, the all2 configuration, every counter). That run splits the gap as:
- **inside** the context 9,315.8 s of cycles: in-context L1D misses 140 per 1,000 instructions, DTLB misses 17.1, issue
  stalls 5,807 cycles per 1,000 instructions. There is **no Plain mcf counter reading on the board**, so the inside gap
  cannot be attributed to the cache, the TLB or the instruction count;
- **outside** 2,258 s over 56,746 exits: 27,481 quantum exits, 28,166 faults, 1,099 system calls, 37,266 CV_STEPs.

## The two arms (C.5a, kit bkit26-gt, module 65ede944a87ea742, 18-bit, counter firmware)

**Board 2, Plain, starts at once.** `505.mcf_r-test-plain-rv64-hpm`: the Plain times stage
(stages-plain-times) with the rusage wrapper and machine-wide counter snapshots around the command, as gcc train's
Plain arm (results/2026-11-05-gcctrain stage-P).

**Board 1, the bundle, after ns-run.sh 1 (pid 1171962) ends with DONE.** `505.mcf_r-test-capstone-lifetimes-devmk-x`:
gtm's mcf stage with machine-wide snapshots around the command and three module parameters set through sysfs for this
stage only, each read back and restored at the end:
- `resume_quantum=1`: a quantum exit resumes inside CV_STEP instead of returning to the launcher;
- `step_budget_ns=10000000000`: otherwise the 80 ms step budget would hand each CV_STEP back to the launcher anyway;
- `retire_exit_sweep=2`: no exit sweep. Safe on silicon by results/2026-11-05-nosweep (N1 holds, Addendum 5). On xz it
  read 8,084.2 s, Plain parity.

**Reference for the bundle: out-gtm-b1**, the same board, kit, bitstream and images, the kit's own parameters. The
bundle changes three switches at once. The exit counters attribute them: CV_STEPs and "between" fall with
resume_quantum, teardown time with the sweep.

## Predictions (`mx-grade.py`)

| # | Prediction | Point estimate |
|---|---|---|
| M0 | rc 0, specdiff PASS, control passes, bench fresh; params; QNT0 007A1200; MX-SET reads 10000000000/Y/2 and MX-RESTORED the kit's 80000000/N/3 | |
| M1 | counters programmed: in-context hpm3-7 in (0, cycles), machine-wide deltas > 0 | |
| M2 | Plain run 8,900-9,270 s | 9,083 s (3 Nov) |
| M3 | bundle run ≤ 0.995 × 11,573.79 s, and CV_STEPs ≤ 15,000 | −90 to −140 s: quantum ×4 removed about 6.1 ms per quantum exit (all-b1 → gtm), of which the launcher round trip is taken as a third to two thirds over the 27,481 left; the sweep about 31 s (174 k pages at xz's 0.178 ms) |

**Reported, not graded:** the Plain mcf counters per 1,000 instructions beside the in-context ones; the bundle's
outside counters (machine-wide minus in-context: instructions, IPC, L1I misses outside the context); inside cycles
against the reference.

**What would refute what.**
- M3 fails on run time with CV_STEPs ≤ 15,000: a quantum exit's cost is not the launcher round trip. It is the context
  switch, the interrupt replay or the cache refill inside, which only a longer quantum removes.
- M3 fails on CV_STEPs: resume_quantum or the budget did not take effect (read MX-SET).

**Grader controls** (perf/mx/ctl):
- out-gtm-b1 given as the bundle: M0 fails (no MX lines), M1 fails (no snapshot), M3 fails: rc 1;
- a mock from it with 11,450 s, 11,000 CV_STEPs, MX lines and snapshots: rc 0;
- gcc train's Plain given as Plain: M2 fails, rc 1;
- an empty folder: rc 2.

**Dry run** (perf/mx/dryrun, every board call stubbed, the real kit, lists and env check): A-ok and B-ok1 DONE;
C-wait (a live pid that then logs DONE) DONE; D-nodone, E-noprereg, G-busy STOP; F-flashfail stops after the retry and
the failed flash back.

**Limits.** N = 1 per arm. Test input. Plain and the bundle run on different boards; the bundle's reference is on its
own board.
