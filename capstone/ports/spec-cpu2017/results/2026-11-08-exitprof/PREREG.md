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

## Addendum 1 (8 Nov 03:4x, before its board call): perlbench c3g with the same two instruments, board 2

Stage `perf/exhpm/stages/500.perlbench_r-test-capstone-lifetimes-devmk-c3g-prof`: c3g's stage, cvprof enabled around
the test command only (the 22,777 s part; makerand runs unsampled). Kit bkit31-exhpm. Board 2, after perlbench
-rg-nd there; it replaces the withdrawn perlbench c3g-thp. The first board use of cvprof is gcc's run on board 3,
an hour earlier.

| # | Prediction |
|---|---|
| Y0 | as X0, for perlbench (specdiff 2/2, 0 FAILED); exhpm lines for all 318 contexts |
| Y1 | run within ±2 % of 33,667.7 s |
| Y2 | ≥ 30 % of the off-CSRUNV samples in the launcher's user code or the delegated system calls' kernel paths |

## Result, gcc on board 3 (8 Nov 04:58): VOID, the core wedged near the command's end

- The control rungs passed with module 16d49244392d22c2, and the control context printed its exhpm line
  (k_in 2,289,548 instructions over a 13.4 M-cycle kernel window: IPC 0.17; between steps 1.12 M in 9.15 M: IPC 0.12).
- The spec step loaded cvprof (`cvprof: loaded`, `@@CVPROF 1`); the stage's heartbeats ran 116 times, about 4,600 s,
  with no new stage output, then stopped. The driver read a wedged core: trap log {seen, mcause} 0x85 (load access
  fault), mepc 0x3fba60df8c (domain VA). No exhpm line for gcc's context, no cvprof dump. **X0 FAILS; X1-X3 VOID.**
- The wedge is not explained. It came at about the time gcc's command ends (4,253 s on c3f). Two variables changed
  at once (the module and cvprof). Board 2's perlbench run with both instruments has passed many context exits with
  cvprof enabled (exhpm lines at 2,621 s), so neither is shown to break a context exit; a split rerun is owed.
