# Where the outside time goes on silicon: gcc train c3f with two instruments (pre-registration)

Written and pushed before the board call (8 Nov, 01:xx). perlbench's gap to Plain is 94 % outside the context (9,800 of
10,391 s); a delegated call costs about 700 k cycles on the board, while QEMU's launcher spends thousands of
instructions on one. Board kernel profiling did not exist: CONFIG_PERF_EVENTS is off, and QEMU's kernel is a different
build (6.1, debug spinlocks, memcg). Two instruments, observation only, in one run:

1. **Module** 16d49244392d22c2 (wt-et perf/exit-hpm 0f0e91e693e8 = c3f's source 3fbed8df1fb8 + the THP commits, off
   + this change, built as 65ede944a87ea742; the control build of 3fbed8df1fb8 reproduces 65ede944's .text, .rodata
   and .data byte for byte). It adds hpm3..8 and instret over the kernel window (CV_STEP minus its CSRUNVs) and the
   between-step window, printed as `capstone_vm: exhpm` per context. Kit bkit31-exhpm = bkit27-gtq with that module.
   QEMU: the windows' instret equals their cycles (k_in 401.5 M / kcyc 401.4 M; u_in 135.1 M / ucyc 135.2 M).
2. **cvprof.ko** (perf/cvprof, 0 two-byte instructions): an hrtimer at the 10 ms tick samples the interrupted pc
   into kernel and user (pc, comm) tables; dumped after the command as `cvprof: k|u count pc symbol`. QEMU: 1,448
   samples, 1,374 at the CSRUNV return (in-context time lands there, interrupts being deferred), the rest named.

**Stage** `perf/exhpm/stages/502.gcc_r-train01-capstone-lifetimes-devmk-c2ac7-prof`: c3f's gcc stage plus two lines
(insmod/enable before the command, dump after it). **Run:** board 3 (perf/exhpm/gt-run-exhpm.sh), after perlbench
c3g-thp there; gcc c3f-thp and gcc -ndcpu move behind it.

| # | Prediction |
|---|---|
| X0 | rc 0, specdiff 1/1, control passes; one exhpm line; a cvprof dump with ≥ 300,000 samples |
| X1 | run within ±1.5 % of 4,253.3 s (the instruments cost < 1 %) |
| X2 | outside windows: IPC (instret / cycles) < 0.2 in the kernel window; the miss counters, not the instruction count, carry most of the kernel window's cycles |
| X3 | the cvprof samples off the CSRUNV return number ≈ outside time × 100/s (739 s → ≈ 74,000) |

X2 is the hypothesis behind "700 k cycles a call": that the exit path stalls rather than computes. Its failure would
point at instruction count, i.e. code to remove.
