# A 4x longer supervisor quantum: gcc train c3f (pre-registration)

Written and pushed before the board call (8 Nov, 11:1x). The exit-window counters (results/2026-11-08-exitprof,
split 2: module 16d49244392d22c2 without cvprof, clean) put gcc's outside time at ~1.5 M cycles per CV_STEP (kernel
window 7.58 G and between-step window 10.01 G cycles over 11,253 steps, IPC 0.15). mcf on 2 MiB pages spends 2,060 s
outside over 27,686 quantum exits, about 74 ms each; its in-context time per quantum, 331 ms, is the firmware's
supervisor quantum (CAPSTONE_SUPERVISOR_QUANTUM=8000000 cycles = 320 ms, results/2026-11-05-outside-levers; the
csupquantum CSR is 32 bits).

**Question.** Is an exit's cost fixed per exit (then a 4x quantum removes about three quarters of the quantum
exits' outside time) or proportional to the time spent masked (HZ_PERIODIC catches up every missed tick at the exit,
and the total would not move)?

**Kit** bkit32-q32 = bkit27-gtq (c3f's: module 65ede944a87ea742, launcher 1d8b4f0829bef0e7) with one firmware
define changed: `-DCAPSTONE_SUPERVISOR_QUANTUM=32000000` (1.28 s); the boot quantum stays 8,000,000. **Stage** c3f's
unchanged gcc train stage (perf/ga2). **Run:** board 3, perf/exhpm/gt-run-q32.sh (gt-run-ga2.sh with the kit).

| # | Prediction |
|---|---|
| Q0 | rc 0, specdiff 1/1, control passes; the boot's QNT0 reads 01E84800 (32,000,000) |
| Q1 | quantum exits (ev kind 1) ≤ 2,500 (c3f 8,327) |
| Q2 | if the cost is per exit: outside ≤ 600 s (c3f 739.5 s) and run ≤ 4,120 s; if per masked time: outside within ±5 % of 739.5 s. Either reading decides the lever for mcf and perlbench |
