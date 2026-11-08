# The unify gcc train runs' extra outside time: the board harness's heartbeat, reading a 760 KB step output every 30 s

Desk reading of existing transcripts (8 Nov); no board call. Sibling record: ACCOUNTING.md (the per-exit kernel work).

## The question

gcc train01 reads 4,518.6-4,611.0 s on the unify stack (out-nd-b2, out-nd3-b3, out-un-b3) against c3f's 4,253.3 s
(out-ga2-b3), with fewer exits and fewer inside cycles. The excess sits in `ucyc`, the module's window from one CV_STEP's
return to the next one's start: 3.08-3.27 M cycles a step against 0.87-1.10 M (results/2026-11-07-nodiag). QEMU reads
~0.1 M for both, so it is board-only. Ruled out before this record: the grant log, the exit digest, console traffic.

## Ruled out here: the launcher's sched_yield

Both launchers' step loops are the same at kind 1. c3f's quiet launcher (wt-fa perf/quiet-grants 51c16d32059c,
exec.c `virtual_service_loop`) and dev's (wt-unify exec.c ~1438-1446) both do
`if (step.kind == 1) { sched_yield(); continue; }`; dev's adds only two relaxed atomic counter increments before it.

## What the per-task deltas show (out-nd3-b3, the only unify gcc run with @@CPU/@@PST snapshots around the command)

- `/proc/stat` cpu over the command: user +46,741 ticks, system +402,573, idle +0, softirq +1,175 (450,489 ticks over
  4,529.6 s).
- The launcher's own times arrive as the stage shell's (pid 107) child times: cutime +155, cstime +372,858. So the
  launcher itself spent 1.6 s in user mode.
- **pid 97, the heartbeat subshell** (`( i=0; while ...; do sleep 30; ... echo "@@HB ..." ...; done ) &` in the
  batch's gate.sh, forked from gate.sh pid 85 before the step): its children's times rose by **cutime +44,567,
  cstime +25,905 ticks, 704.7 s of CPU**, 95 % of every user tick on the machine.
- The heartbeat's body (gate.sh line 14):
  `echo "@@HB $i out=$(wc -c < /tmp/step.out)B/$(wc -l < /tmp/step.out)L tail=$(tail -c 240 /tmp/step.out | base64 | tr -d '\n')"`.
  Busybox `wc` reads the whole file through its per-character loop, twice.
- **What is in /tmp/step.out:** the step's whole stdout so far. The unify gcc stages print the kernel's text symbols
  first (`echo @@KSYMS-BEGIN; grep -E ' [tT] ' /proc/kallsyms; echo @@KSYMS-END`, 19,983 lines), so the first heartbeat
  already reads `out=762587B/19983L`. c3f's stage (perf/ga2) has no dump: `out=538B/6L`.
- So each heartbeat costs about 6.8 s of CPU on the 25 MHz core (704.7 s over about 103 complete heartbeats in the
  window, per the auditor's decoding of their tails), and the heartbeat loop's period stretches to ~43 s, against
  34.5-34.8 s on every small-file stage (c3f gcc, c3g and unify perlbench).
- **Control before the command.** At @@PST0 no launcher has run yet, and pid 97's children already read cutime 773,
  cstime 623 (14.0 s) for the first two heartbeats: ~7.0 s each with no context window at all, so replayed ticks are
  not what inflates the figure. Tick conservation agrees: the launcher's 373,013 ticks cover its inside time (3,432 s)
  plus `kcyc - rcyc` (195 s) with ~30 s to spare, so at most ~30 s could have been misattributed.
- **An instrument that does not count ticks agrees.** Machine-wide instret (GT-SNAP hpm) outside the context is 8.08 G
  (nd3) and 8.00 G (nd-b2) against c3f's 2.79 G (ga2): 5.2-5.3 G more, about 33 instructions per byte over ~103 x 2
  x 770 KB, a getc loop (busybox 1.36.1 `wc` reads every byte through `getc` for both -c and -l).
- It lands in `ucyc` because the launcher only gives the hart up between steps: a woken heartbeat process sets
  `need_resched`, CV_STEP's stop check returns to the launcher, and the heartbeat runs before the next CV_STEP.

| gcc train01 run | step.out at HB 1 | ucyc | per step | exits kind 1 / 3 |
|---|---|---|---|---|
| out-un-b3, out-nd-b2, out-nd3-b3 (unify) | 762,587 B | 770.3 / 767.1 / 767.3 s | 3.08-3.27 M | 10,485-10,495 / 527-528 |
| out-ga2-b3, out-split2-b3, out-fill-b2, out-q32-b3, out-thp3-b3 (c3f) | 387-538 B | 388.0-435.5 s | 0.77-1.10 M | 1,302-8,331 / 7,525-7,528 |

The table alone is confounded (every dump run is a unify run). What breaks the confound: the per-task CPU above, read
directly, and the unify **perlbench** runs, which have no dump (out-un-b3 and out-nd-b2: `out=226B` / `241B` at HB 1)
and whose total `ucyc` is lower (4,214 s and 4,191 s against 4,946-5,493 s on the three devmk perlbench runs,
fit-perlbench.txt). Per event it is higher (37.6 ms per service exit against 18.7 ms on best-b2; 0.75 M against
0.29-0.46 M cycles per step), so a per-step cost of the unify stack is NOT excluded by perlbench; for gcc, with 527
service exits, it could be small. QEMU shows none of it because the
QEMU path (virtualvm.py) runs no heartbeat.

## How big the fix is: a band, not a point

- The excess to explain, without the exit digest: outside +344 s (nd-b2, 1,083.5 against 739.5 s) and +358 s (nd3);
  `ucyc` about +375 s. The headline ~440 s was un-b3's, which still carried the 92 s digest.
- **Upper:** 704.7 s, the heartbeat children's measured CPU in the dump run.
- **Lower:** ~370 s, unify's `ucyc` excess over c3f's (767 against ~400 s).
- That the heartbeat accounts for the excess (and not only for ~705 s of CPU) is PLAUSIBLE BUT UNPROVEN: it needs
  c3f's small-file heartbeat to cost ~2.8 s each, or unify's other outside work to be cheaper than c3f's (plausible
  with 527 against 7,528 service exits). The small-file cost has never been measured (out-nd3-b3 is the only board
  run with @@PST snapshots, N = 1); PREREG-hbfix.md measures it. unify serves 527 calls a run against
  c3f's 7,525, which should leave its `ucyc` well below c3f's once the heartbeat is cheap; the difference between the
  two bounds is the heartbeat's cost on a small file (six process spawns every 30 s), which no run measured (no c3f
  or Plain run carries @@PST snapshots).
- If it holds, unify gcc train would read about 3,830-4,150 s from 4,518.6 s (nodiag), against c3f's 4,253.3 s.

## The fix (harness, not kit; for the stage owner)

- Write the kallsyms dump to a file outside /tmp/step.out (or print it in a step of its own); and/or
- make the heartbeat O(1): sizes from `stat -c %s` / `ls -l`, no `wc -l`.
- The pre-registration of the fixed run should carry the @@CPU/@@PST snapshots, so the heartbeat's own cost on a small
  file is read once: that number also bounds what every other board stage pays.

**Not affected:** the running SPEC train lists. None of the 17 stages in stages-train-fair prints kallsyms (0 matches of
`KSYMS|kallsyms` in each run.sh, 8 Nov).

## A prior record's mechanism, corrected (claim-auditor, 8 Nov)

2026-11-06-unify's result says its tick probe was VOID because "the console was still draining the stage's kallsyms
dump". That mechanism cannot be right: gate.sh sends the step's stdout to /tmp/step.out and the console gets it only at
`cat /tmp/step.out` after the step. HB 1's tail in all three unify gcc runs ends in `@@TICKPROBE-BEGIN`: the probe ran
beside the heartbeat's `wc` over the 762 KB file (three readings, share 0.324-0.325, ~9.7 s of 30 s). The VOID verdict
stands; the stated cause was the heartbeat. Likewise 2026-11-07-nodiag's reading that context time shows up as user
time is contradicted by the launcher's cutime of +155 ticks: the 467 s of user time is the heartbeat's children.
