# The unified stack without the exit digest: gcc train and perlbench on board 2 (pre-registration)

Written and pushed before the board calls (7 Nov, 18:0x). The unified stack's outside-time regression is the
launcher's SHA-256 of the whole image at every context exit, switched on by the batch harness's
CAPSTONE_EXEC_DIAGNOSTICS=1 (results/2026-11-06-unify, last section). The lead chose this over the second perlbench
THP run on board 2 (results/2026-11-07-thp2: that run is withdrawn before any board call; board 3's stays).

**Stages.** The two unify board stages (perf/unify/bstages: gcc -rgz, perlbench -rg), changed by one line before the
arm: `unset CAPSTONE_EXEC_DIAGNOSTICS; echo "@@NODIAG ${CAPSTONE_EXEC_DIAGNOSTICS-unset}"`. Same images, kit
bkit30-unify, parameters. QEMU (gcc): launcher instructions in the run window 1.352 G → 0.002 G, specdiff PASS,
outputs equal the earlier twin's. A fault in these runs prints no cause line (the switch gates it).

**Runs:** board 2 (perf/unify/un-run.sh 2: flash C.5a, the list, flash bitstream 3 back), after the fill run there
ends with DONE: gcc train -rgz-nd, then perlbench -rg-nd.

**References** (board 3): gcc unify 4,611.0 s (inside 3,432.1, outside 1,179.0, ucyc 19.26 G cycles); c3f 4,253.3 s
(outside 739.5 s). perlbench unify 37,131.8 s (inside 21,901.2, outside 15,230.5); c3g 33,667.7 s (outside 9,799.8 s).

| # | Prediction |
|---|---|
| N0 | rc 0, specdiff PASS (perlbench 2/2, test.out 0 FAILED), control passes, outputs equal the twins'; @@NODIAG reads unset; the .err files hold no `exit image` line |
| N1 | gcc: inside within ±1 % of 3,432.1 s; outside ≤ 900 s; run ≤ 4,253.3 s (c3f); point 4,100-4,250 s (1.20× Plain is 4,218.6 s) |
| N2 | perlbench: inside within ±1.5 % of 21,901.2 s; run ≤ 33,667.7 s (c3g); point 30,500-32,500 s (1.31-1.39× Plain) |

**What would refute what.** N1 holding with N2 failing: perlbench's excess outside time has a second cause beyond the
digest (its exits differ from gcc's: 316 contexts, delegated system calls). N1's outside bound failing: the digest
was not the whole of gcc's 440 s.

## Result, gcc (7 Nov 19:25): 4,518.6 s, N0 holds, N1 FAILS

- **N0 holds:** rc 0, specdiff 1/1 PASS, outputs equal the twin's, control passes; @@NODIAG reads unset; no `exit image` line.
- **N1 FAILS:** inside 3,435.1 s (13.426 G, +0.1 % on unify's 3,432.1 s: holds); outside 1,083.5 s against ≤ 900 s;
  run 4,518.6 s against ≤ 4,253.3 s. 1.287× Plain on board 3's 3,510.0 s.
- Removing the digest saved 92 s of unify's 4,611.0 s. ucyc is unchanged (19.18 G against 19.26 G cycles; c3f 9.84 G),
  so the digest was not the per-step excess (results/2026-11-06-unify, RETRACTION).
- perlbench -rg-nd runs next on board 2. There the digest ran at each of 318 context exits; N2 is still open.

## Addendum 1 (7 Nov, 21:1x, before its board call): gcc -nd again, with CPU-time snapshots, on board 3

The per-step excess of the unified stack (ucyc 19.2 G against c3f's 9.8 G cycles) is board-only: on the board the
machine executes 6.7 G more instructions outside the context, on QEMU fewer. Ruled out from source: the quantum
sched_yield (c3f's launcher has it too) and module timers or workers (neither module has any). Work that grows with
wall time fits; who does it is unknown.

**Stage** `perf/nodiag/bstages/502.gcc_r-train01-capstone-lifetimes-unify-rgz-ndcpu`: the -nd stage plus `cpusnap`
before mark1 and after mark2: the `cpu` line of /proc/stat and `/proc/<pid>/stat` of every process. QEMU: rc 0,
specdiff PASS, 43 processes per snapshot, the context's time reads as system time (30 → 1,178 ticks).

**Run:** board 3, after gcc c3f-thp (results/2026-11-07-thp2, G1), which runs after perlbench c3g-thp there.

| # | Prediction |
|---|---|
| D0 | as N0, plus both snapshots present |
| D1 | run within ±1 % of 4,518.6 s (board 2); the snapshots attribute ≥ 80 % of the outside time (≈ 1,080 s) to named processes and /proc/stat states |
| D2 | which one carries the ~380 s excess over c3f is not predicted: the run exists to name it |

## Result, perlbench (8 Nov 04:04): 30,120.3 s, N2 holds; the best perlbench so far

| perlbench test | run | inside | outside | exits [quantum, service, fault] |
|---|---|---|---|---|
| c3g (board 2) | 33,667.7 s | 23,867.9 s (100.53 G) | 9,799.8 s | 19,264 / 265,988 / 84,937 |
| unify (board 3) | 37,131.8 s | 21,901.2 s | 15,230.5 s | |
| **unify -nd (board 2)** | **30,120.3 s** | 21,869.3 s (93.55 G) | 8,251.0 s | 52,285 / 112,220 / 22,967 |

- **N0 holds:** rc 0, specdiff 2/2 PASS, test.out 317 ok / 0 FAILED, outputs equal the twin's; @@NODIAG unset; no
  `exit image` line. **N2 holds:** inside −0.1 % on unify's; run below c3g's, and below the 30,500-32,500 s point band.
- **The digest was perlbench's excess.** 7,011.5 s went with it, 22.0 s per context exit over 318 contexts. The
  module's windows did not move (ucyc 4,247 s against 4,272 s, kcyc 2,870 against 2,859 s): the digest ran after each
  context's last step, outside every window, as the RETRACTION above says for gcc, where one context paid 92 s.
- **perlbench is 1.294× Plain's 23,277.1 s** (c3g: 1.446×). Plain with the mapping cache (results/2026-11-07-plainreuse)
  is the fair denominator for a stack that caches mappings; its perlbench run is on board 1 now.

## Result, gcc -ndcpu on board 3 (8 Nov 07:5x): D1 holds; the snapshots cannot answer D2

- 4,529.6 s (+0.24 % on board 2's 4,518.6 s), inside 3,432.1 s, outside 1,097.5 s; specdiff 1/1 PASS, outputs equal the
  twin's. **D1's run clause holds.**
- /proc/stat over the command: user 467.4 s, system 4,025.7 s, softirq 11.8 s, idle 0. Processes alive at both
  snapshots: rcu_sched 29.9 s, the batch shell 20.0 s, kworker mm_percpu_wq 9.3 s; the launcher is born and dies
  between the snapshots and is not in them.
- **D1's attribution clause FAILS and D2 is not read:** the user/system split is not trustworthy here. QEMU puts the
  launcher at about 15 M instructions for this run without the digest, which 467 s of user time cannot be. The
  ticks a context misses while interrupts are masked are accounted at once, in the mode the interrupt is finally
  taken in; when that is the launcher's user mode after the ioctl returns, context time reads as user time.
- The same bias applies to cvprof's samples: deferred ticks land at the first point interrupts are taken, which
  must be separated when its perlbench dump is read (results/2026-11-08-exitprof).

## Correction (8 Nov 22:1x): the 467 s of user time is the heartbeat's children, not context time

The reading above, that deferred ticks make context time read as user time, is withdrawn, and so is its extension to
cvprof's samples. The launcher's cutime moved only +155 ticks. The 467 s of user time is the board heartbeat's
children: `wc -c` and `wc -l` over a step output that begins with a 762 KB kallsyms dump (perf/exit-kernel
90b69152e63f, results/2026-11-08-exitk/UNIFY-440S.md). D1's attribution clause still FAILS, for this reason instead.
The digest result of this record (perlbench 37,132 → 30,120 s) is unaffected. Both perlbench arms' step outputs stay
under 70 KB, so neither carries the large-file heartbeat cost.
