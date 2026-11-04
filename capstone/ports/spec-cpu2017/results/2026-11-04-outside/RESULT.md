# Time outside the context: work or waiting? Result

The boot ran on board 2 (ObjBounds 603528b8b, kit bkit15-objb-b2), booking 658, 15:10-15:21 on 4 Nov. It was pre-registered at 276f7a5dedba, with Addendum 1 at 5ed42bf48d5f. Stage `stages-plain-times/538.imagick_r-test-capstone-spatial-times`.

**Verdict.** The hart never idles during the run. Hypothesis A (waiting) is refuted. By the letter of the thresholds the reading is O (another task takes the hart). That reading is mostly an instrument defect, described below. After correcting for it, the outside time is CPU work counted to this run, nearly all in kernel mode. One other task stands out: `rcu_sched`.

## Readings, run command

| Reading | Value |
|---|---|
| T1 | control PASSED; rc 0, specdiff PASS, outputs equal to the twin's; instret 108,724,432 (reference 108,723,327, +0.001 %); cycles 386,252,237 (−0.39 %) |
| VM's own window (`elapsed_ns`) | 30.80 s (reference 30.87 s): the command itself is unchanged |
| real (up1 − up0) | **76.82 s** (reference 33.09 s), see the defect below |
| children user / sys (`times` delta) | 5.70 s / 41.01 s |
| `/proc/stat` delta: user / system / **idle** / softirq | 7.94 s / 47.77 s / **0** / 0.14 s; total 55.85 s |
| other tasks (per-task delta) | `rcu_sched` 2.73 s (all system); `kworker/0:1` 0.17 s; the snapshot's own sh/awk 1.4 s; everything else under 0.05 s |

## The instrument defect

run.sh takes the second snapshot (`times`, `/proc/stat`, the per-task dump) between the command and `u1=$(up)`. So "real" includes one whole snapshot, and the per-task dump forks a `cat` for each of 36 tasks.
- **The snapshot cost.** By the VM's elapsed time it costs about 22 s: 76.8 − 33.1 ≈ 2 × 21.9 s. The first snapshot also lies inside real.
- **Why the counters are still usable.** The `times` and `/proc/stat` readings are taken before each dump. So their delta covers the command plus the first dump only.
- **The cross-check.** The `/proc/stat` delta (55.85 s) equals the command (33.1 s) plus one dump (about 22 s).

**The correct reading.**
- Every tick of that window is user or system time, and idle is 0.
- 46.7 s of it is the shell's children: the launcher, plus the first dump's `cat` and `awk`.
- About 3 s is other tasks, `rcu_sched` most of all.

The pre-registered thresholds (real − (U + S), I) assumed a real window without snapshots, and that does not hold here.

## What this settles and what it does not

- **Settled.** Outside time is not the hart waiting: idle is 0. It is CPU work in kernel mode, during the run's own exits. That is the module, Linux and the monitor's M-mode switch, which Linux counts as the launcher's system time. Fewer exits (fault-ahead) and cheaper exits are therefore the right levers.
- **New.** `rcu_sched` took 2.7 s of system time in about 55 s, roughly 5 %. On an otherwise quiet single-hart board that is large. Its cause is not identified.
- **Open.**
  - Why an exit costs 10-15 ms of kernel CPU when the twin counts about 15 k kernel instructions per exit. The candidates are kernel CPI after each context switch (caches and TLB flushed), and M-mode monitor time.
  - The hpm counters, which have no privilege filter, could read it if they were also read around the outside window.
- **Repeat.** A repeat should put the snapshots outside the timed window, take one snapshot only, and drop the per-task dump. That dump has done its job: no task other than `rcu_sched` takes the hart.
