# gcc train (train01): resume_quantum, a 10 s step budget and the quiet-grants launcher: pre-registration

Written and pushed before the board call (6 Nov, 00:35). Driver `perf/gx/gt-run.sh`: perf/gtm/gt-run.sh with the kit
and launcher hash replaced. Grader `gx-grade.py`, stage `stage-x/run.sh` (written by `mk-stage.py`), list `b2.list`.

## Why

gcc train is 1.58× Plain (results/2026-11-05-gcctrain), and outside time is half the gap: 11.3 ms over each of
155,699 exits. The QEMU symbol profile (perf/xprof, kit-profe: Capstone minus Plain outside instructions) puts
11.6 k extra instructions on each exit. Two of the sources need no new module:
- **The launcher's grant log.** `grant()` in exec.c prints a `CAPSTONE_VM_ADD_OK` line for every admitted grant:
  26,297 lines, each an fprintf and an unbuffered write(2). The launcher's printf frames are 50 M instructions
  (`__printf_buffer`, `_write`, `strchrnul`, `_itoa_word`), plus the kernel's write path.
  - New build option `-DCAPSTONE_VEXEC_QUIET_GRANTS` (wt-fa perf/quiet-grants 51c16d32059c): only slot 0, the image
    grant that the board parsers read, is printed unless CAPSTONE_VM_TRACE is set.
  - Board vexec 1d8b4f0829bef0e7, QEMU 9251005fcc197743. With the option off the build is byte-identical to today's
    0bde5fd8e0c9c374. A rebuild of the old commit reproduces 0bde5fd8 and a7596c8e, so the build is reproducible.
- **Quantum exits back to the launcher.** `resume_quantum=1` and `step_budget_ns=10000000000`, set through sysfs for this
  stage only, read back and restored. The sweep stays at the kit's 3. Mode 2 would send every in-run CV_RETIRE to the
  load-and-store sweep (results/2026-11-05-mcfx Addendum 1).

**QEMU gate** (perf/xprof gtrX, kit-profq):
- rc 0, specdiff PASS, one ADD_OK line in the .err;
- outside instructions: launcher 170.8 → 86.0 M, kernel 2,068 → 1,994 M, module unchanged;
- the gate ran with the sweep at 2. That changes no output and does not touch the launcher.

## The run

Board 2, after perf/mx's Plain mcf (driver pid 1370451) ends with DONE. Kit bkit27-gtq: bkit26-gt (C.5a, module
65ede944a87ea742, 18-bit, counter firmware, all2 params) with the new launcher. The control rungs run first in the
boot, as in every stage.

**Reference: out-gt-b2**, 5,562.1 s. Same board, bitstream, module, firmware and images, with the old launcher and the
kit's own parameters. That makes three changes against it. The exit counters separate them:
- CV_STEPs and "between" fall with resume_quantum;
- the .err line count shows the quiet launcher.

## Predictions (`gx-grade.py`)

| # | Prediction | Point estimate |
|---|---|---|
| M0 | rc 0, specdiff PASS, control, bench fresh, kit params, QNT0 007A1200; MX-SET reads 10000000000/Y/3, MX-RESTORED 80000000/N/3 | |
| M1 | in-context hpm3-7 in (0, cycles), machine-wide deltas > 0 | |
| X2 | the .err holds exactly one CAPSTONE_VM_ADD_OK line | 1 (reference 26,297) |
| M3 | run ≤ 0.995 × 5,562.1 s and CV_STEPs ≤ 75,000 (reference 77,246) | −40 to −75 s. Quiet grants: about 130 M fewer instructions at outside CPI 7.6 is about 40 s. resume_quantum: 4,248 quantum exits at a few ms each is 10-15 s |

**What would refute what.**
- M3 fails on run time with X2 holding: the grant log was not a cost on silicon. A write to the tmpfs .err is cheap
  there, and QEMU's instruction count overstated it.
- X2 fails: the kit did not carry the launcher.

**Grader controls** (perf/gx/ctl):
- out-gt-b2 given as the run: M0, X2 and M3 fail, rc 1;
- a mock with 5,450 s, 72,000 CV_STEPs, MX lines and a one-line .err: rc 0;
- an empty folder: rc 2.

**Dry run** (perf/gx/dryrun, board calls stubbed; the chain stub now applies spec-board.sh's stage-path check):
- A-wait (a live pid that then logs DONE): DONE, the stage path checked;
- B-nodone: STOP.

**Limits.** N = 1. The quiet launcher has not booted on silicon before this run; the control rungs precede the stage.
