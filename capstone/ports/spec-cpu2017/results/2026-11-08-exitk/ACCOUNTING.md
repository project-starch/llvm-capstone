# The Linux work around a context exit on silicon: accounting from the existing board transcripts (8 Nov)

No board call was made for this record. Every number is read from transcripts already on disk under
`$CAPSTONE_WORK/spec/board/out-*/<stage>/spec.txt` (module `capstone_vm: bench|exits|exhpm` lines, GT-SNAP, @@CPU/@@PST).
Scripts beside this file: `ctxparse.py` (per-context lines to JSON), `fit.py` (least squares over contexts),
`xcheck.py` (out-of-sample check on single-context runs). Outputs: `fit-perlbench.txt`, `xcheck.txt`.
The core clock is 25 MHz (GT-SNAP cycle delta 106.17 G over split2's 4,256.3 s = 24.94 MHz); one 10 ms tick period is
250,000 cycles.

## 1. The kernel window costs 13.5-14.4 k cycles per 10 ms the context ran masked, plus ~30 k per CV_STEP

The module's exit timing splits every CV_STEP ioctl into the CSRUNV windows (inside), `rcyc` (in-loop fault
resolution) and the rest of the ioctl, `kcyc - rcyc`. Fitting `kcyc - rcyc` over perlbench's 317 contexts per run
(the makerand context left out), with the masked tick periods `inside cycles / 250,000` as one regressor:

| perlbench test run (board) | R² | per CV_STEP | per masked 10 ms |
|---|---|---|---|
| out-best-b2 c3g (2) | 0.9996 | 32.2 k | 13.55 k |
| out-rest-b1 ac (1) | 0.9992 | 22.5 k | 13.97 k |
| out-un-b3 unify-rg (3) | 0.9998 | 28.0 k | 13.65 k |
| out-thp2-b3 c3g-thp (3) | 0.9995 | 33.3 k | 13.56 k |
| out-nd-b2 unify-rg-nd (2) | 0.9998 | 27.9 k | 13.60 k |

Adding the quantum-exit count as a regressor moves the masked-time coefficient to 13.9-14.6 k (fit-perlbench.txt).

**Out of sample** (xcheck.txt), `30 k x steps + 13.8 k x inside/250,000` against the measured `kcyc - rcyc` of every
single-context run, none of which was fitted:

| run | inside | measured / predicted | the masked-time term |
|---|---|---|---|
| gcc train c3f, out-ga2-b3 / out-split2-b3 / out-fill-b2 | 3,514 s | 1.008 / 1.015 / 1.008 | 194 s |
| gcc train unify, out-un-b3 / out-nd-b2 / out-nd3-b3 | 3,432 s | 0.999 / 0.990 / 0.992 | 189 s |
| gcc train dcnofp, out-dcnofp-b3 (32 M quantum) | 3,253 s | 0.952 | 180 s |
| gcc train c3f, 32 M quantum, out-q32-b3 | 3,504 s | 0.949 | 193 s |
| mcf test, out-mx-b1 / out-gtm-b1 / out-thp-b3 | 9,170-9,316 s | 1.003 / 0.966 / 0.960 | 506-514 s |
| x264 test c3f, out-rest-b1 | 38,799 s | 0.987 | 2,142 s |
| xz test, out-ns-b2x | 266 s | 0.940 | 15 s |

So the model predicts the kernel window of gcc, mcf, x264 and xz to 1-6 % without being fitted to them. The two runs
at the 32 M quantum read 5 % low: some of the term is per interrupt, not per period (results/2026-11-08-quantum:
7,025 fewer quantum exits, kcyc − rcyc 5.228 → 4.838 G, about 56 k cycles each).

**What the term is.** The board kernel is `CONFIG_HZ_PERIODIC=y`, `CONFIG_HZ=100`, `# CONFIG_HIGH_RES_TIMERS is not set`
(build-fpga/build/linux-6.4.14/.config) on the oneshot-only RISC-V timer (`timer-riscv.c:55` `CLOCK_EVT_FEAT_ONESHOT`).
The context runs with interrupts off (`local_irq_save` around `run_event`, capstone_vm.c CV_STEP loop); the pending
timer interrupt is taken at the `local_irq_restore` right after the CSRUNV, inside the kernel window. Its handler,
`tick_handle_periodic` (kernel/time/tick-common.c), runs `tick_periodic()` once, then for a oneshot device loops
`next = ktime_add_ns(next, TICK_NSEC); if (!clockevents_program_event(dev, next, false)) return; ... tick_periodic(cpu);`
— one full tick (do_timer(1), update_wall_time, update_process_times: scheduler_tick, run_local_timers, RCU) for
every 10 ms the context was masked. C.5a has no `time` CSR: each clock read in that path (`ktime_get` in
`clockevents_program_event`, the clocksource read in `update_wall_time`, `sched_clock` in `scheduler_tick`,
`hrtimer_run_queues`) is an illegal-instruction trap the monitor emulates (sbi_capstone.c `CSR_TIME` emulation; the
rtl lane's tickcost record). 13.8 k cycles is about 2.2 k instructions per replayed period at the window's IPC 0.16.
The stock board kernel's kallsyms (out-nd3-b3 spec.txt) has `tick_periodic` and `tick_handle_periodic` and no
`tick_nohz_handler`.

The accounting is consistent with that: on out-nd3-b3 the `/proc/stat` cpu line advanced 450,489 ticks over 4,529.6 s
of command (99.5 % of wall x HZ), 402,573 of them system: every masked period is replayed and charged to the launcher.

**What the data cannot separate.** The regression cannot tell the replay itself from other kernel work that also grows
with masked time and runs at the same interrupt: timer callbacks whose expiry fell inside the masked window (timer
softirq), RCU softirq work, and the M-mode firmware time (the hpm counters have no privilege filter). By source those
are per expired timer or per grace period, not per period; how much of the 13.8 k they are is UNRESOLVED.

## 2. Per step and per exit, by window

| | gcc train c3f, split2-b3 (11,174 steps, 18,704 entries) | perlbench test c3g, best-b2 (281,332 steps, 368,243 entries) |
|---|---|---|
| inside (CSRUNV) | 3,513.6 s | 23,316 s |
| kernel window minus faults (`kcyc - rcyc`) | 210.4 s; masked-time term 194 s, per-step term ~13 s | 1,670 s; masked-time term 1,287 s, per-step ~340 s |
| fault resolution (`rcyc`) | 92.5 s | 1,562 s |
| between steps (`ucyc`) | 400.0 s (0.90 M cycles/step) | 4,969 s (0.44 M cycles/step) |
| other ioctls (`ocyc`), teardown, launch | 16.6 + 13.4 + 6.0 s | 893 + 203 s + launch |
| outside total | 742.7 s | ~9,700 s |

gcc's GT-SNAP deltas over the command (split2): `/proc/stat` intr 71,371 (6.4 per step, 3.8 per CSRUNV entry), ctxt
60,722 (5.4 per step), `/proc/vmstat` pgfault 141,679 (12.7 per step). The exhpm module (same run) reads 1.217 G
instructions at IPC 0.160 in the kernel window and 1.455 G at IPC 0.145 between steps.

**Interrupts per exit** cannot be separated by source: `/proc/stat` intr is a total (timer, UART, PLIC), and one
timer interrupt carries a whole replay, so the interrupt count says nothing about ticks.

**The between-step window** (`ucyc`) is the larger part for perlbench and gcc. It does not fit a per-step plus
per-masked-period model well (R² 0.77-0.86; the masked-time coefficient swings 21-89 k across runs), so no
per-masked-time cost there is claimed. It holds the launcher's delegated calls (kfast's lever, C1), whatever runs when
the launcher yields after `need_resched` (kernel threads woken by the replay, the board harness's heartbeat), and, in
the unify gcc runs, a board-only harness cost (UNIFY-440S.md beside this file).

**Other tasks** (out-nd3-b3, @@PST deltas over the command): rcu_sched 2,988 ticks (29.9 s, 0.66 %), kworker
mm_percpu_wq 928 (9.3 s), ksoftirqd 63. The harness heartbeat's children 70,472 (UNIFY-440S.md). Nothing else above 5.

## 3. The levers, ranked by predicted saving

| # | lever | mechanism | predicted saving | state |
|---|---|---|---|---|
| K1 | board kernel `NO_HZ_IDLE` (HZ_PERIODIC off; HIGH_RES_TIMERS stays off) | the tick runs as `tick_nohz_handler`: `tick_do_update_jiffies64` adds `ticks += ktime_divns(delta, incr)` and `jiffies_64 += ticks` once (tick-sched.c:103-118), one `update_process_times`, and `hrtimer_forward(&ts->sched_timer, now, TICK_NSEC)` skips the missed periods (tick-sched.c:1390). The replay loop is gone | up to the masked-time term less one tick per timer interrupt: gcc train ~150-190 s of 4,253 s (3.5-4.5 %), mcf test ~400-500 s, perlbench test ~1,000-1,280 s, x264 test ~1,700-2,100 s | **built, kit bkit35-nohz** |
| K2 | HIGH_RES_TIMERS + NO_HZ_IDLE (QEMU's own setup) | the same O(1) catch-up through `tick_sched_timer`, plus hrtimer interrupt overhead | no more than K1 | not built: cannot beat K1 |
| K3 | a lower HZ | fewer periods to replay | none available: HZ_100 is Kconfig's lowest | — |
| K4 | NO_HZ_FULL | needs a housekeeping CPU; this board has one hart | — | — |
| M1 | module: one clock read per entry (`cv_step_stop`'s `ktime_get_ns`) | 1-2 emulated rdtime traps per entry | entries x trap cost: under ~5 s for gcc | not built: too small |
| M2 | module/launcher per-step work (`ucyc`) | kfast (C1, another lane) | perlbench: its own estimate | other lane |
| H1 | board harness: the heartbeat re-reads the whole step output twice every 30 s | UNIFY-440S.md | unify gcc train: 370-700 s; every other stage: unmeasured | harness fix, not a kit |

Fault resolution (`rcyc`) is pinning: ~1.1 ms a page on this core (results/2026-11-05-outside-levers), and the
zeroing Plain also pays; THP cut mcf's 393 s to 91 s already (out-thp-b3). Not a per-exit Linux cost.
