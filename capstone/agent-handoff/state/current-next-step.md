# Current recommended next step

## 2026-07-30 (03:30) — SQLite on silicon: one variable left

**C-13 IS FIXED.** The descriptor-driven entry glue runs on hardware. `beebs_primer1`
with real interp returns its oracle 582955588. Every SQLite dependency is now verified
on silicon *individually*:

| dependency | evidence |
|---|---|
| entry glue reads its descriptor | `beebs_primer1` real interp, oracle |
| zero-fill / bulk copy / byte tail / >2040 B / private `.L` | `gpsz gpcp gptl gpbg gppv`, each its own oracle |
| 2 MiB domain creation | `bigwin` FAIL→PASS across the `data_off` fix |
| large code window (0x140000) | `bigwin` carries SQLite's exact window |
| this firmware boots and runs domains | `beebs_primer1`, 03:22 |

**SQLITE STILL HANGS**, and this is now a clean measurement — single session under the
board lock, firmware proven minutes earlier, domain content-hash matched:

    booted to root shell, /dev/capstone present
    on-board domain 1623008 == local build
    Ok, good file. / Found 2 segments / Entry address = 10000
    Globals offset = 0x140000 / Loadable size = 1389480
    -> hangs, 30-minute timeout

It dies after the ELF parse, somewhere in domain creation or the first call (the two are
indistinguishable from the UART: the kernel module's pr_info does not reach the console,
and `capstone_error` does C_PRINT then `while(1)` with C_PRINT going to the RTL trace, so
every monitor error is a silent spin).

### THE NEXT RUN, already built and QEMU-gated

**`bigmany`** — 64 globals at SQLite's exact 2 MiB allocation (DOMAIN_WINDOW=0x140000),
trivial contents, oracle 2631595461. It fills the one untested cell:

    bigwin    1 global    2 MiB    PASSES
    gpn8..64  many        128 KiB  (QEMU-gated; board run was lost to contention)
    bigmany   64          2 MiB    <-- RUN THIS
    SQLite    1059        2 MiB    HANGS

  * `bigmany` HANGS  -> a count-x-size interaction; SQLite's contents are irrelevant.
    Bisect count downward at 2 MiB.
  * `bigmany` PASSES -> the remaining suspect is SQLite's 78 KB initializer blob
    (bigmany's is ~1 KB). Bisect by growing bigmany's initialized data.

Run it with the launcher, which now holds an exclusive lock:

    LOG=/tmp/bm.log bash capstone/tests/rtl-smoke/run-board-ladder.sh bigmany

### HARD-WON PROCESS RULES (each cost a session on 2026-07-29/30)

1. **ONE board session at a time.** Three concurrent runners power-cycled each other
   mid-JTAG-load and produced a bootrom loop that looked exactly like a corrupt
   firmware — including a control rung "failing to boot". That false signal triggered a
   firmware rebuild hunt which then broke a *working* image. `run-board-ladder.sh` now
   takes a non-blocking flock; the SQLite runner still needs wrapping in the same lock.
2. **Never rebuild an artifact a live session depends on.** Rebuilding firmware under a
   running board session killed it mid-load.
3. **Do not "fix" infrastructure on the critical path.** The payload-ordering fix
   addressed a problem that was provably not affecting any result (verified by content
   hash) and broke the boot. Note the defect, add a detector, fix it after the
   measurement is in hand.
4. **Compare artifacts by CONTENT, never size.** The stale and current SQLite domains are
   both 1,623,008 bytes.
5. **Absolute paths after any `cd`**, and never `pgrep -f` a pattern that appears in your
   own command line — both bit repeatedly.

### STILL OPEN, lower priority

- `gpstress` (6 mixed globals, 128 KiB) returns wrong data on silicon while each of its
  five paths passes alone. Same "works singly, fails combined" shape as SQLite; may be
  the same bug at small scale, which would make it a much cheaper reproducer.
- The count bisection at 128 KiB (`gpn8/16/32/64`) was lost to contention; rerun if
  `bigmany` implicates count.
- UART transfers: fast tier fails its sha 100% of the time. Root-caused to the POLLING
  transport dropping whole POST batches; runs now prefer the venv interpreter with
  websocket, but **that fix is UNTESTED** — do not claim it works until a session shows
  `burst=16` succeeding first time.
