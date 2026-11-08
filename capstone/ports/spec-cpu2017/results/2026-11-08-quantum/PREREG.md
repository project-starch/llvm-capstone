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

## Result (8 Nov 12:16): 4,215.0 s (−38 s); quantum exits fall 6.4x, outside falls 3.9 %

| gcc train c3f, board 3 | run | inside | outside | kcyc / rcyc / ucyc / ocyc (s) | exits [quantum, service, fault] |
|---|---|---|---|---|---|
| quantum 8 M (6 Nov) | 4,253.3 s | 3,513.8 s | 739.5 s | 306 / 97 / 394 / 14 | 8,327 / 7,528 / 2,875 |
| **quantum 32 M** | **4,215.0 s** | 3,504.4 s | 710.6 s | 283 / 89 / 388 / 23 | 1,302 / 7,525 / 2,883 |

- rc 0, specdiff 1/1 PASS, control passes.
- **Q0 FAILS on its readback clause only:** the boot's QNT0 reads 007A1200 because QNT0 prints the BOOT quantum
  (unchanged, 8,000,000); the supervisor quantum is set at the supervised call. The quantum exits show it took effect.
- **Q1 holds:** 1,302 quantum exits (≤ 2,500).
- **Q2: the per-exit branch FAILS** (outside 710.6 s against ≤ 600 s); the per-masked-time branch's band (±5 % of
  739.5 s) holds at −3.9 %. Removing 7,025 quantum exits saved 29 s outside (38 s of run): about 4-5 ms each. A
  quantum exit under resume_quantum is cheap, and the outside time is not mainly a fixed cost per exit.
- The longer quantum is kept for the following runs as a small, free gain.
