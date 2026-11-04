# Time outside the context: work or waiting? Pre-registration

This file is written and pushed before the boot.

**The question.**
- On SPEC train, about 17 % of every Capstone run is spent outside the program's context, whatever the benchmark:
  - lbm 17.0 %, mcf 17.9 %, imagick 15.8 %, perlbench 17.7 %;
  - nab test 16.6 %, x264 test 16.7 %.
- That is 10-15 ms per context exit.
- On the QEMU twin, the extra kernel instructions per exit are far fewer. On imagick test, the kernel runs 36 M instructions in the Spatial run against Plain's 15 M: about 15 k per exit over 1,402 exits. At 25 MHz, 12.6 ms is about 315 k cycles.
- So either the work outside runs at a CPI near 20, or the hart is not working on this run for much of that time.
- These need different fixes, so this boot separates them.

**The boot.**
- **Stage.** `spec/stages-plain-times/538.imagick_r-test-capstone-spatial-times`: board 2's imagick test Spatial stage with an unchanged plan.json and the same images.
- **What run.sh adds.** Before and after each command it prints the shell's `times` (user and system CPU time of the waited-for children, at HZ=100 tick accounting) and the first line of `/proc/stat`.
- **Parsing.** A transcript with these lines injected parses to the same result.json as the original (run-stage.py read).
- **Where it runs.** Board 2, ObjBounds 603528b8b, kit bkit15-objb-b2. It goes in after the running xz devmk stage and before the mcf evaluation pair.
- **Reference.** out-objb/538.imagick_r-test-capstone-spatial: run 33.09 s, inside 15.51 s, outside 17.58 s. Its VM stats: launch_ns 5.94 s and elapsed_ns 30.87 s.

**Readings, for the run command:** real = up1 − up0; children's user U and system S from the `times` delta; idle I from the `/proc/stat` delta (idle + iowait, ticks).

| # | Prediction or reading |
|---|---|
| T1 | The control passes. The run is rc 0, specdiff PASS, and its outputs equal the twin's. Cycles and instret are within 2 % / 0.1 % of the reference. |
| T2 | Reported: U, S, I and real, in seconds. |
| W (work) | If real − (U + S) < 2 s and I < 2 s, the outside time is this process's CPU work: kernel, module and launcher. The fix is fewer exits or cheaper ones. |
| A (away) | If I ≥ 5 s, the hart idles for that long during the run, waiting on something. The fix is in what it waits for: the exit or re-entry path, a sleep, a tick. |
| O (other) | If real − (U + S) ≥ 5 s and I < 2 s, another task takes the hart. |

Readings between these thresholds are reported, and no reading is chosen.

**Limits.**
- N = 1.
- Tick accounting has a resolution of about 10 ms per sample, which is fine over 33 s.
- Linux may not see time spent in the monitor in M mode as either user or system. Such time would show up as neither, without being idle.
- If I is small but real − (U + S) is large, that is the case: the reading does not separate it from another task, and it says so.

## Addendum 1 (4 Nov 12:40, before the boot)

**Two findings since the pre-registration.**
- **Plain loses time outside its own process too.** On the Plain test `times` stages, real − (user + sys) is 9.1 % on nab, 9.1 % on lbm, 9.0 % on xz and 20 % on imagick.
  - So part of the Capstone outside time is not Capstone's.
  - Against Plain's real − user, the Spatial outside excess on test is +12 % (nab), +17 % (lbm), +27 % (mcf), +43 % (xz) and +57 % (imagick) of Plain's run time.
- **The launcher calls `sched_yield()` at every supervisor-quantum exit** (`exec.c` virtual_service_loop, kind 1). If any other task is runnable on the one hart, the hart goes to it.

**One added reading.** Beside each `times`, run.sh now prints every task's utime and stime (`/proc/<pid>/stat` fields 14 and 15) as `@@TASK <pid> <comm> <utime> <stime>`. The before/after delta names the task that takes the hart if O is read.

The thresholds above are unchanged. A transcript with these lines injected still parses to the same result.json.
