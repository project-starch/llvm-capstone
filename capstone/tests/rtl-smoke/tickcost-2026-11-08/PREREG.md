# Pre-registration: where the hart is when a cvprof run wedges (8 Nov 2026, before any board run)

## Retraction (this branch only, 9e73913e0cac)

The first version of this file proposed a "tick-replay livelock": a tick callback longer than TICK_NSEC = 10 ms
makes `tick_handle_periodic()` loop forever. **That is wrong by source.** `hrtimer_run_queues()` runs a timer only
when the clock has passed its expiry (`__hrtimer_run_queues`, `basenow` vs `softexpires`). cvprof's
`hrtimer_forward_now(t, 10 ms)` puts the next expiry 10 ms after the callback ends. So a slow callback runs at most
once per replay, the following replays are cheap, and the loop re-arms and returns. A long callback slows the
kernel; it cannot hang it. Arms C and D now predict RETURN. The withdrawn version was never run.

## What the three cvprof wedges show (raw transcripts)

Runs: `$CAPSTONE_WORK/spec/board/out-xp-b3`, `out-xp2-b2`, `out-split1-b3` (`guest/board.txt`). Each run's last
"vbatch wedge apertures" block reads:

- **Trap log (aperture 255) = 0x85: a timer interrupt, not a load access fault.**
  - The log keeps mcause[6:0] only (C.5a 1fc2cd5aa `core/cva6.sv:1311`), and filters only causes 0 and 2 (`:1191-1193`).
  - The tval is an instruction encoding: `0x318a3683` = `ld a3,792(s4)`, `0x7b783` = `ld a5,0(a5)`,
    `0x1913c23` = `sd s9,24(sp)`. A store cannot raise cause 5. The decoder writes the instruction bits into tval
    before it marks an interrupt (`decoder.sv:1966-1968`, `TvalEn = 1`).
  - So the last logged trap is an STI (0x8000000000000005) at mepc.
- **Where mepc points.**
  - perlbench: 0x2aca438fa0, the launcher.
  - gcc: 0x3fba60df8c and 0x3fad876ef0. These are 291 MB and 105 MB above the runs' domain image bases
    (0x3fa90a5000, 0x3fa7402000; image 405,504 bytes), so they are not domain code. The vDSO is the likely
    executable mapping there (UNRESOLVED without a map).
  - So all three ticks landed in user code, as a real trap.
  - The log is commit's exception output after the escape strip (`cva6.sv:2084`), and a context runs with
    interrupts masked (entry: mie, mideleg, mip = 0, mstatus.MIE = 0; the module enters under `local_irq_save`).
- **Nothing unfiltered was logged after that STI.**
  - The next MTI would log 0x87, the tick's re-arm ecall 0x89, and any page fault 0x8c/0x8d/0x8f.
  - OpenSBI masks MTIE when it forwards a timer interrupt, and only `sbi_set_timer` (the re-arm in
    `tick_handle_periodic`, after `tick_periodic()` and thus after cvprof's callback) unmasks it.
  - So the hart never completed the tick it entered with that STI, and never left it through an unfiltered cause.
    `rdtime` IS filtered: C.5a has no `CSR_TIME`, so every clock read is an illegal-instruction trap (cause 2) that
    M-mode firmware emulates.
- **Aperture 224 = 0xff in all three: {ex_commit.valid, load/store syncer set, ready x3, flush, privM}.**
  - An exception is committing at M at the sampled instant. pb10 and pb10-fix6b, which followed a kernel Oops and
    panic, read 0xff too, as does fso-fix6.
  - `ex_commit.valid` is a single-cycle signal when traps are ordinary. Six of six samples at 1 means either a trap
    on (nearly) every cycle (a handler whose first instruction traps again with a filtered cause) or an exception
    held at commit and never taken.
  - The commit-pc aperture reads 0xffffffff817ffffe (both gcc runs), 0xffffffff813ffffe (perlbench), and
    0xffffffff803ffffe (pb10): always ...ffffe. Recorded, not understood.
- **No panic, Oops, BUG or RCU line in any of the three transcripts.**
- **The context itself is untouched.** perlbench with (out-xp2-b2) and without (out-best-b2) cvprof, 205 contexts
  matched in order: the context cycle-count ratio has median 1.000 (5th-95th percentile 0.998-1.004). Wall-clock lag
  grows at a rising rate: 2.8 % over base 839-1,217 s, 5.1 % over 2,586-3,216 s, 7.9 % over 4,251-5,011 s, 8.1 % over
  8,477-9,070 s. That is the profiler's per-tick cost growing as its tables fill. It is not shown to cause the wedge.

## What this kit asks

What does the board read when the hart is stuck inside the tick handler for a known reason? And does cvprof's
full-table path, or a long callback, stop the kernel by itself?

| arm | what | predicted reading |
|---|---|---|
| A | `scanbench`: three full miss-scans of each cvprof-shaped table, irqs off | six `tickcost: scan` lines, probes 8192 (user) and 16384 (kernel); `@@TC A ALIVE`. The ns values are the measurement |
| B | callback spins 3,000 us per tick, 30 s | returns: `calls` about 2,900-3,000, `maxbody_us` 3,000-3,500 |
| C | cvprof's tick body, verbatim, over FULL tables (every tick a full scan), 30 s | returns: `kfull + ufull = samples`, `maxbody_us` about arm A's scan time. If it does NOT return, cvprof's full-table path stops the kernel by itself: the finding |
| D | callback spins 15,000 us per tick, 30 s | returns, heavily slowed: `calls` below 3,000, `maxbody_us` 15,000-15,600 |
| E1 (boot 1, last) | callback never returns, looping on `ktime_get` (one emulated `rdtime` trap per pass) | no `@@TC E1 ALIVE`; `tickcost: stuck rdtime` printed; trap log frozen at 0x85 (the STI that entered the tick), tval decoding as an instruction. 224: privM mostly 1, `ex_commit.valid` mostly 0 (firmware runs many instructions per trap) |
| E2 (boot 2, last) | callback never returns, register loop, no traps | no `@@TC E2 ALIVE`; `tickcost: stuck register`; trap log 0x85 as E1. 224: privM 0 and `ex_commit.valid` 0 |

**For the wedge arms, the board lane is asked to read aperture 224 ten times, and the commit-pc and trap-log mepc
twice.** One sample cannot tell "an exception every cycle" from "an exception sometimes".

## Grading

- **VOID:**
  - insmod fails or arm A does not print six lines;
  - B does not return;
  - or the boot dies before the E arm starts (unless C or D is what stopped: then that is the finding).
- **E1/E2:**
  - Each confirms the trap-log reading if the log freezes at 0x85 with an instruction tval.
  - If 224 reads 0xff in most samples in E1 or E2, the cvprof signature is just "stuck in the tick", the RTL is not
    implicated, and the next step is finding the stopping point in cvprof's path.
  - If neither reproduces 224 = 0xff, the cvprof wedges are not a plain software spin. A trap storm or a held
    exception is a core-level state, and the next step is RTL: what makes `ex_commit.valid` hold at M.
- **C or D does not return:** the slow-tick path itself stops this kernel, against the source reading above.

## Ordinary (no-sampler) runs

Not answerable yet: the stopping point is not known. Known so far: the capstone_vm module registers no hrtimer,
timer, tasklet or irq (dev and spec-silicon, by grep), so nothing it adds runs inside the tick. The six runs without
cvprof did not wedge.
