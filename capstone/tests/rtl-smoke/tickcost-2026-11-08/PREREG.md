# Pre-registration: a tick callback longer than one period livelocks the board kernel (8 Nov 2026, before any board run)

## What the three cvprof wedges show (read from the raw transcripts)

Runs: `$CAPSTONE_WORK/spec/board/out-xp-b3`, `out-xp2-b2`, `out-split1-b3` (`guest/board.txt`). Each run's last
"vbatch wedge apertures" block reads:

- **Aperture 255 (trap log) = 0x85. This is an interrupt, not a load access fault.**
  - The log keeps mcause[6:0] only (C.5a 1fc2cd5aa `core/cva6.sv:1311`), and filters only causes 0 and 2 (`:1191-1193`).
  - The latched tval is an instruction encoding in all three runs:
    - `0x318a3683` = `ld a3,792(s4)`
    - `0x7b783` = `ld a5,0(a5)`
    - `0x1913c23` = `sd s9,24(sp)`
  - A store cannot raise cause 5. The decoder writes the instruction bits into tval before it marks an interrupt
    (`decoder.sv:1966-1968`, `TvalEn = 1`, config pkg `:70`).
  - So the last logged trap is a supervisor timer interrupt (0x8000000000000005), taken at the interrupted pc (mepc).
- **The log is commit's exception output after the escape strip** (`cva6.sv:2084`; `commit_stage.sv` zeroes
  exception_o under `escape_trap_d | escape_q | dom_switch_busy_i`). An interrupt that escapes a context is never
  logged.
  - Inside a context there are no interrupts anyway: the entry writes the context's mie, mideleg and mip = 0 and its
    mstatus with MIE = 0 (`csr_regfile.sv`, switcher id 69). The module also enters every CSRUNV under
    `local_irq_save` (runtime lane, capstone_vm.c).
  - So the logged STI was taken outside a context, at U level. mepc 0x2aca438fa0 is the launcher. 0x3fba60df8c and
    0x3fad876ef0 are in the 0x3f.. mmap range (runtime lane: probably the domain image, UNRESOLVED).
- **After that STI nothing unfiltered was ever logged again.** Every tick's re-arm (`sbi_set_timer`, an ecall, cause
  9) would log 0x89, every page fault 0x8c/0x8d/0x8f, and every M timer interrupt 0x87. So the hart never again left
  the code it entered with that STI, except through causes 0 or 2. `rdtime` is cause 2 here: there is no `CSR_TIME`
  in C.5a's `csr_regfile.sv`, so every clock read traps to M-mode firmware.
- Aperture 224 = 0xff at the sample ({ex_commit.valid, ..., flush, privM}): an exception committing at M. pb10 and
  pb10-fix6b, which followed a kernel Oops and panic, read 0xff too, as does fso-fix6 (its end not read here). So this
  reads as a "kernel is gone" signature, not an RTL-specific one.

## The proposed mechanism (kernel software, no RTL defect needed)

1. The board kernel is `CONFIG_HZ_PERIODIC=y`, `HZ=100`, no `HIGH_RES_TIMERS`, a oneshot-only clockevent
   (`timer-riscv.c`: `CLOCK_EVT_FEAT_ONESHOT`, no `FEAT_KTIME`). Read from
   `build-fpga/build/linux-6.4.14/.config` and its source.
2. `tick_handle_periodic()` (`kernel/time/tick-common.c`) runs `tick_periodic()`. It then loops
   `next += TICK_NSEC; if (!clockevents_program_event(dev, next, false)) return; tick_periodic();`.
   `clockevents_program_event()` returns -ETIME WITHOUT calling `set_next_event` (no SBI ecall) while `next` is
   already in the past. If one `tick_periodic()` costs more than TICK_NSEC = 10 ms, the loop never ends: a silent
   livelock in hardirq. It makes no ecall and no page fault, and traps only on `rdtime`.
3. Without high-resolution timers, cvprof's hrtimer callback runs inside `tick_periodic()`
   (`update_process_times` -> `run_local_timers` -> `hrtimer_run_queues`). It probes a 256 KB open-addressing table
   with linear probing and no probe limit: user 8192 x 32 B, kernel 16384 x 16 B. Once the table is full (or nearly
   full), an absent key scans the whole table. In the replay loop every iteration samples the SAME interrupted pc.
4. The core runs at about 20.8 MHz (split 2: 87.84 G cycles over 4,222.5 s), so one period is about 208 k cycles.
   A full scan is 8-16 k iterations over 4,096 cache lines that miss L1D. That is plausibly more than one period.
   Arm A measures it.
5. The kernel stays silent: `SOFTLOCKUP_DETECTOR` is off. The RCU stall warning (`TREE_RCU`, 21 s) is `pr_err`, and the
   console runs at loglevel 1 during these runs.

**Consistent with it:**
- QEMU passes, because its host-speed scan is far below 10 ms of guest time.
- The wedges come late in long runs, after a table has had time to fill.
- perlbench matched context by context, with (out-xp2-b2) and without (out-best-b2) cvprof:
  - every context's own cycle count is unchanged: the ratio over all 205 contexts has median 1.000, 5th-95th
    percentile 0.998-1.004, extremes 0.986 and 1.033;
  - wall-clock lag grows across the run at a rising rate: 2.8 % over base 839-1,217 s (contexts 1-33), 5.1 % over
    2,586-3,216 s (49-65), 7.9 % over 4,251-5,011 s (97-121), 8.1 % over 8,477-9,070 s (185-201);
  - that is the profiler's per-tick cost growing as its table fills.

**Not shown yet:**
- that cvprof's tables actually reached full in those runs (no dump survived);
- that a full scan costs more than 10 ms on silicon.

Arms A and C answer the second. A disable-time `ufull`/`kfull` count on a real run would answer the first.

## The arms (one boot, `tickcost-guest.sh`, in this order; only D is expected not to return)

| arm | what | predicted reading |
|---|---|---|
| A | `scanbench`: three full miss-scans of each table, irqs off | six `tickcost: scan` lines, probes 8192 (user) and 16384 (kernel); `@@TC A ALIVE`. The ns values are the measurement |
| B | callback spins 3,000 us per tick, 30 s | returns: `tickcost: disable mode=0 calls=` about 2,900-3,000, `maxbody_us` 3,000-3,500; `@@TC B ALIVE` |
| C | cvprof's tick body, verbatim, over FULL tables (every tick a full scan), 30 s, console at loglevel 8 | decided by arm A's KERNEL scan (ticks land mostly in the idle kernel). Above 9,500,000 ns: no `@@TC C ALIVE`, the run wedges as in D. Below 7,000,000 ns: returns with `maxbody_us` about the scan time and `kfull + ufull = samples`. In between: no prediction |
| D | callback spins 15,000 us per tick, 60 s, console at loglevel 8 | does NOT return: no `@@TC D ALIVE`, no disable line, heartbeats stop, the driver reads a wedge with trap log 0x85 whose tval decodes as an instruction. Not graded: an `rcu: INFO: rcu_sched self-detected stall` print about 21 s in, with `tick_handle_periodic` in its trace (only if a grace period is pending); apertures 224/221/222 recorded beside the cvprof wedges' |

## Grading

- **VOID:**
  - insmod fails;
  - arm A does not print six lines;
  - arm B does not return (then even a short callback hangs this kernel, and the threshold story is wrong);
  - or the boot dies before arm C starts.
- **MECHANISM CONFIRMED:** B returns AND D wedges with trap log 0x85 (or C wedges first, with the same log). Then a
  tick-path callback longer than one period livelocks this kernel with the cvprof wedges' signature.
  - If C also wedges, or arm A's scan exceeds 10 ms, cvprof's own full-table tick crosses the line.
- **REFUTED:** D returns, i.e. the replay loop is bounded by something not read here. The cvprof wedges then need
  another explanation, and the RTL interrupt/escape sim becomes the next step.

## Ordinary (no-sampler) runs

Mechanism 2 needs one `tick_periodic()` above 10 ms. In the stock tick the only per-period work is the scheduler,
timekeeping and RCU, plus a few `rdtime` emulations. The capstone_vm module registers no hrtimer, timer, tasklet or
irq (dev and spec-silicon, by grep). `timer_list` callbacks run in softirq, after the replay loop has exited. So
ordinary runs cannot reach this livelock unless something adds that much hardirq-time work per tick.
