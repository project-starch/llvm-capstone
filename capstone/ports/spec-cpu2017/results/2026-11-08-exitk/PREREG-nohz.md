# The tick replay removed: gcc train c3f on a NO_HZ_IDLE board kernel (DRAFT pre-registration)

DRAFT, written 8 Nov before any board use; the boards are booked for SPEC train until Wednesday and nothing here runs
before then. The predictions below are fixed now; the run date and board are the coordinator's.

## Why

ACCOUNTING.md (beside this file): the kernel window outside the context, minus fault resolution, costs ~30 k cycles
per CV_STEP plus 13.5-14.4 k cycles per 10 ms the context ran masked, on five perlbench runs and, out of sample, on
gcc, mcf, x264 and xz to 1-6 %. By source that second term is the periodic tick's replay: the board kernel is
HZ_PERIODIC on a oneshot-only timer, and `tick_handle_periodic` runs `tick_periodic()` once per missed period after
every masked CSRUNV, each with emulated `rdtime` traps. For gcc train c3f it is 194 s of 4,253 s.

Under `CONFIG_NO_HZ_IDLE` (low-resolution nohz: HIGH_RES_TIMERS stays off) the same interrupt runs `tick_nohz_handler`:
`tick_do_update_jiffies64` adds all missed periods in one step (`ticks += ktime_divns(delta, incr)`, `jiffies_64 +=
ticks`, linux-6.4.14 kernel/time/tick-sched.c:103-118), then one `update_process_times` and
`hrtimer_forward(&ts->sched_timer, now, TICK_NSEC)` (:1390). The switch happens at the first `hrtimer_run_queues`
because the RISC-V clocksource is continuous (made VALID_FOR_HRES without a watchdog, clocksource.c:694-697), the
clockevent is oneshot, and `nohz` is on by default (the board's command line, `earlycon console=ttyS0,57600`, does not
turn it off).

## What changes: one variable, the kernel

- **Kit bkit35-nohz** = `cp -rL` of bkit27-gtq (C.5a 1fc2cd5aa; module 65ede944a87ea742; launcher 1d8b4f0829bef0e7;
  control module f2a2645cc570dcc6; firmware defines, params and driver unchanged), link count 1 on every file,
  SHA256SUMS regenerated. Two edits, both bkit29-thp's pattern:
  - board-run.sh relinks the private kernel tree perf/exitk/linux-board-nohz on the bake's own rootfs.cpio and
    replaces the Image (the shared Image kept as Image.stock). Its guard requires `CONFIG_NO_HZ_IDLE=y`,
    `# CONFIG_HZ_PERIODIC is not set`, `# CONFIG_HIGH_RES_TIMERS is not set` and `CONFIG_HZ=100` in that tree (it
    passes on the nohz tree and stops on the stock tree); a relink must rebuild `usr/initramfs_data.o`.
  - bake-manifest.sh saves the bake's rootfs.cpio.
- **Kernel**: a copy of build-fpga/build/linux-6.4.14 with `scripts/config -d HZ_PERIODIC -e NO_HZ_IDLE`, olddefconfig:
  config-board-nohz.diff (NO_HZ_IDLE, NO_HZ_COMMON, TICK_ONESHOT on; NTP_PPS's "not set" line gone). Its Image before
  relink: df5ce98ef886988a. A standalone relink of the block, cut from the kit's board-run.sh, on the stock cpio seed:
  `nohz kernel 4dc22f6f58076b3a`, initramfs rebuilt. System.map: `tick_nohz_handler` present (stock: absent; the stock
  board kallsyms in out-nd3-b3 lists `tick_handle_periodic` and `tick_periodic`).
- **Why the modules are unchanged.** c3f's source (3fbed8df1fb8) built with 65ede944's recipe against the stock tree
  reproduces 65ede944's .text, .rodata and .data; built against the nohz tree it is identical to 65ede944 in all 36
  sections but `.rodata.str1.8` (the source path) and the build-id. The control module rebuilt against the nohz tree is
  f2a2645cc570dcc6 byte for byte. vermagic `6.4.14 SMP riscv` in both.
- **Stages** (perf/exitk/stages): `gtr-c3f-tk` = perf/ga2's c3f gcc train stage, and `plain-gtr-tk` = perf/gt's Plain
  gcc train stage, each with three observation-only lines outside the timed command: `@@TL` (the tick device's
  `event_handler` from /proc/timer_list, once) and `@@TK0`/`@@TK1` (/proc/stat cpu, intr, ctxt and uptime, just inside
  `u0=$(up)` and `u1=$(up)`).

## QEMU (what it does and does not measure)

QEMU's own kernel is already NO_HZ_IDLE + HIGH_RES_TIMERS at HZ=250, and its quantum is counted on the cycle CSR, which
follows the icount clock: at `-icount shift=5` the gcc twin reads 84,679 quantum exits against 1,330 at shift 0
(perf/best gtr-c3f), so a masked window stays under ~10 ms of guest time whatever the shift. The twins never replay
more than one period and cannot show the saving at HZ=100. Two QEMU 6.1 kernels on the board's tick setup (HZ=100, no
HIGH_RES_TIMERS), `qb` HZ_PERIODIC and `qn` NO_HZ_IDLE, test function and the readback; a pair at HZ=1000 (8 periods per
8 ms quantum) tests the mechanism. Results: QEMU.md beside this file.

## The run

Board 3, C.5a, one boot each, in order: (1) `gtr-c3f-tk` (step 14,400 s, booking 150 min), (2) `plain-gtr-tk` (7,200 s,
75 min). Driver perf/exitk/board/gt-run-nohz.sh (perf/b3/gt-run-ga2.sh with the kit, a kit check for the relink block and
the tree's config, and the pre-registration branch as a parameter), list perf/exitk/board/b3.list. The control rungs run
first in each boot. Dry run: perf/exitk/board/dry.

**References** (same board, bitstream, module, launcher, firmware and stage; stock kernel): Capstone out-ga2-b3, 4,253.3 s
(inside 3,513.8 s; `kcyc - rcyc` 5.228 G cycles; `ucyc` 9.84 G; steps 11,253; exits 8,327 / 7,528 / 2,875); Plain
out-fill-b3 (perf/gt Plain stage on bkit27-gtq), 3,510.0 s.

| # | Prediction |
|---|---|
| N0 | rc 0, specdiff 1/1 PASS, output hashes equal out-ga2-b3's; control rungs pass; the kit log reads `nohz kernel <hash>`; `@@TL` reads `event_handler: tick_nohz_handler`. **If `@@TL` reads `tick_handle_periodic`, the run is VOID** (the kernel never left periodic mode) |
| N1 | `kcyc - rcyc` ≤ 2.2 G cycles (reference 5.228 G); point 0.8-1.6 G |
| N2 | `/proc/stat` cpu ticks over the command (TK1 - TK0, all fields) ≤ 30 % of (up1 - up0) x 100; point 12-20 %. Stock kernels account 99.5 % (out-nd3-b3) |
| N3 | inside cycles within ±0.5 % of 87.85 G; in-context instret within ±0.01 % of the reference |
| N4 | run ≤ 4,133 s (reference − 120 s); point 4,060-4,110 s |
| N5 | `ucyc` within ±10 % of 9.84 G (393 s) |
| P1 | Plain gcc train within ±1.5 % of 3,510.0 s; `@@TL` reads `tick_nohz_handler` |

**What each reading would mean.**
- N1 and N4 hold: the replay was the masked-time term, and the kernel change is the lever for every benchmark in
  proportion to its inside time (predicted savings in ACCOUNTING.md §3).
- N1 holds, N4 fails: the replay's cycles reappeared elsewhere (deferred timer or RCU work after the exit); read N5.
- N1 fails with N0 holding: the masked-time term is not the replay but per-period work that NO_HZ keeps (expired
  timer callbacks, RCU); the lever is void and the term needs another instrument.
- N5 falling by more than 10 % says the between-step window also carried replay-driven work.
- P1 failing: the kernel change moves Plain, and Plain must be re-measured on it before any ratio is quoted.

## Optional second board (board 2, if time allows): perlbench test c3g

Stage perf/exitk/stages/perl-c3g-tk (perf/best's c3g stage with the same three lines), same kit, against out-best-b2
(test command 33,024 s; `kcyc - rcyc` over its 317 contexts 1,670 s, masked-time term 1,287 s): test command
−1,000 to −1,300 s; `kcyc - rcyc` ≤ 600 s; specdiff 2/2.

## Limits

N = 1 each. QEMU cannot measure the saving at the board's HZ. NO_HZ_IDLE also stops the tick in idle; idle is 0 % during
these runs (out-nd3-b3 /proc/stat), so it should not show. Tick-based CPU accounting will under-count system time on the
new kernel (one tick per exit): stime-based readings from these runs are not comparable to earlier runs.
