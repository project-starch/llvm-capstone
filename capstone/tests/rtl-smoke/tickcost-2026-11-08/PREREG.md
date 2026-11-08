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
  - gcc: 0x3fba60df8c and 0x3fad876ef0, in the 0x3f.. mmap range where the launcher maps the domain image.
  - WITHDRAWN (8 Nov, same day, before any run): an earlier version said these lie 105-291 MB above "the domain image".
    That base (0x3fa90a5000, 405,504 bytes) is the CONTROL step's ADD_OK line (out-xp-b3 board.txt:107, inside
    `@@STEP-BEGIN control`), not gcc's. gcc's image base is unknown, so these may be gcc domain code: UNRESOLVED.
  - The log is commit's exception output after the escape strip (`cva6.sv:2084`), and a context runs with interrupts
    masked (entry: mie, mideleg, mip = 0, mstatus.MIE = 0; the module enters under `local_irq_save`).
  - If mepc is domain code, then a logged interrupt at a domain pc means a domain instruction reached commit with an
    interrupt marked while supervision was off. That is an RTL-level anomaly.
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
- **The machine was static at the wedge reads.** The driver reads every aperture twice, 0.6 s apart, and prints
  UNSTABLE on a change. All three cvprof wedges have ZERO UNSTABLE lines over about 25 apertures, the 8-byte commit
  pc included. The check fires: out-hfl's heartbeat-limit stop, with its kernel alive, has 12, among them
  224 0x7f -> 0xff.
  - out-hfl's live system also reads 220 0xcf, 221 0xfe, 222 0x6e, 193 7, 194 3, the same as the cvprof wedges. So
    those LSU/tag values are not wedge evidence, and 224 = 0xff also occurs on a live system.
  - The E arms calibrate stability: how many UNSTABLE lines a software stall inside the tick produces.
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

## Result (board 3, C.5a 1fc2cd5aa, 8 Nov; transcripts sha256 f1805cca75d0fdfb (E1) and aa581864848cf32a (E2))

The runtime lane ran both boots (kit bkit33-tickcost, tickcost.ko c16c5c56585b84c1, driver with
`BAKED_WEDGE_REPEAT_224=10`). They were graded with `tc-grade.py`, fixed first (see below): **both rc 0, AS
PRE-REGISTERED** on the graded items.

| arm | pre-registered | E1 boot | E2 boot |
|---|---|---|---|
| A | six scan lines, probes 8192/16384 | user 18.87-18.90 ms, kernel 37.66-37.67 ms | user 18.68-18.71 ms, kernel 37.65-37.66 ms |
| B | returns, maxbody 3,000-3,500 us, calls about 2,900-3,000 | returns, maxbody 3,106, **calls 1,500** | returns, 3,107, **1,500** |
| C | returns, maxbody about the scan, kfull + ufull = samples | returns, 37,695 us, 633 = 551 + 82 | returns, 37,686 us, 630 = 558 + 72 |
| D | returns, maxbody 15,000-15,600, calls below 3,000 | returns, 15,106, 1,521 | returns, 15,106, 1,512 |
| E | stuck; trap log 0x85, tval an instruction | E1: 0x85, tval 0xa0d63 (a branch), mepc 0xffffffff8002fdea | E2: the same |
| E 224 (reading) | E1: privM mostly 1, ex_commit.valid mostly 0; E2: both 0 | **E1: 0xff on 11/11 reads, no UNSTABLE** | E2: 0x1c on 11/11, as predicted |

**Two predictions MISSED:**
- B's call count. A low-res hrtimer forwarded 10 ms from the end of its callback expires on the tick after next,
  so 20 ms per call.
- E1's 224. The rdtime-emulation calibration in simulation (`ex_commit.valid` 0/75) does not predict silicon
  (11/11), and is withdrawn as a reference. The reason for the difference is not known.

**What this settles:**
- A software loop stuck inside the tick handler that reads the clock (E1) reproduces the cvprof wedges' signature
  exactly: trap log 0x85 with an instruction tval, 224 = 0xff on every read, zero UNSTABLE, and a commit pc ending
  in ...fffe (0xffffffff80f6fffe).
- **RETRACTED:** the reading that the cvprof wedges are a frozen hart, an RTL state. It was sent to the runtime lane
  this afternoon and recorded on this branch at 06119da2cced.
- A full-table cvprof tick (arm C, 37.7 ms per tick, every tick) slows this kernel but does not hang it in 30 s.
- The stopping point in the real cvprof runs is a loop inside the tick path that reads the clock. It is NOT
  identified.

**Grader fix made before grading:**
- Arm completion now comes from the kernel's enable/disable console lines, paired in order. The @@TC markers come
  only through heartbeat tails, which stopped at arm C's start.
- The disable lines were keyed by mode, which merged B and D.
- Re-tested on seven synthetic captures, the two real boots, and a missing file.

## Addendum 2 (8 Nov, before its board call): validate a GDB reader on a known stuck loop

`wedge_gdb.py` halts the hart over the board's GDB session after a wedge is declared, samples the pc 8 times across
resume/halt, and reads ra, sp, mcause, mepc, mtval, mstatus, scause, sepc and the disassembly. It never resets. It
was dry-run on stub consoles: values in order; a dead session gives UNREAD on every line; the session always stops.
`tickcost-guest.sh` now prints `tickcost-addr tick|cvprof_body|ktime_get 0x...` at KERN_EMERG before the arms.

**The validation boot:** tickcost E2 (the register loop `.L85`, tick + 0x154 .. tick + 0x160, three instructions),
with the driver calling `probe()` after the wedge apertures.
- **READER VALID:** all 8 pc samples in [tick + 0x154, tick + 0x160), using the `tick` address the boot printed, and
  the disassembly shows `addi a5,a5,1` / `addi a5,a5,1` / `j`.
- **READER INVALID:** any pc sample reading 0xca11ab1ebadcab1e or outside that range, or UNREAD.

Optional second boot, E1 (`.L83`, tick + 0x170 .. tick + 0x17c). Pre-registered: samples in that range, in
`ktime_get`'s body, or in M-mode firmware [0x80000000, 0x80200000) (the rdtime emulation), and mstatus.MPP
consistent with each.

Only after READER VALID does a real cvprof wedge get the reader (the runtime lane's call, 1.5-5 h of board time).
