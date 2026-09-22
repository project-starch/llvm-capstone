# Silicon-ladder revival — pre-registration, written before the first boot (2026-09-22)

## What this run is

`fpga-silicon-measurements-for-paper.md` §2 publishes **eight** ladder rungs and, at `:260-277`,
**eight more that do not appear**, each with *"a clean, correct baseline half"*. Six of the eight are
blocked on **R-1**, which `ISSUES-ARCHIVE.md:25` records as **`GONE 2026-09-05`** — *"all three probes
read as the reference model"*. R-1's own impact line names four of the casualties directly:
*"`matmult_int`, `coremark_matrix`, `beebs_crc32`, `beebs_insertsort` unmeasurable."*

**The table has not been re-run since that blocker was retired.** It is also the only section of the
measurements doc still standing on a bitstream and a compiler that no longer exist; everything else is
on `caplifive_m1_054cea69b`.

This run asks one question: **how many of the eight are measurable now, and does the published table
still hold on current silicon and current compiler?**

## Vehicle, and why this runner and not the faster one

`run-board-ladder.sh` → `run_ladder_perf_fpga.py`, which **power-cycles per rung**.

The faster baked path (`run_baked_rungs_fpga.py`) was rejected on evidence, not preference. It
*"calls cold_boot ONCE and then runs every rung from that single boot, which is exactly the R-3
condition"*, and instructs: *"Build each rung at a distinct DOMAIN_BASE_VA (0x10000, 0x20000, ...) or
expect a silent hang that looks just like a rung result."*

**Measured before choosing, and a first reading of it CORRECTED.** Invoked standalone,
`build-ladder-fpga.sh` leaves every rung at entry point `0x10000` — all 14 identical. I first read that
as `H1-platform.md:145-146` being false. It is not. The board driver's own `preflight_artifacts()`
rebuilds every artifact with `LADDER_DISTINCT_VA=1` and relocates each rung to its own base — observed
in this run's log as `relocated domain base VA -> 0x20000 … 0x60000`, one per rung after the first.
The distinct-VA guarantee is real; it lives in the driver's preflight, not in the build script alone.

What follows for the runner choice is unchanged, and is the reason the faster path is still rejected:
`run_baked_rungs_fpga.py` is a *different* driver and does not perform that relocation, which is
exactly why its own docstring tells you to build each rung at a distinct `DOMAIN_BASE_VA`. Baking the
artifacts `build-ladder-fpga.sh` produces standalone would put every rung at `0x10000` in one boot —
the R-3 condition, and a silent hang that reads as a rung result.

## Desk gates already passed (no board time)

- Both halves built from `ladder-rungs.spec`, which exists so the two halves cannot drift to different
  `-O` levels. **`LADDER_OPT` was not passed.**
- Baseline half: `ladder_base_ctl`, **`capability-instructions=0 (must be 0)`** — the build's own check.
- **QEMU parity, exit 0, all 14 rungs matched their oracles** via `run-ladder-perf-qemu.sh`, which runs
  *"the SAME freestanding controller the board runs … exact parity with `run_ladder_perf_fpga.py`,
  minus the hardware"*. A rung failing here would not have been sent to the board.
- Three oracles reproduce the 2026-07-28 record exactly — `rv8_sha512 = 1390718314`,
  `rv8_sha512s = 2842840124`, `beebs_ns = 1184999093` — so the build is at the same parameterisation
  the published table used.

## Native oracles, frozen before the run

| rung | oracle | standing |
|---|---:|---|
| `ctrsanity` | 43260934 | anchor / control |
| `beebs_prime` | 582955588 | anchor |
| `beebs_aha_mont64` | 2185097489 | anchor |
| `matmult_int` | 774662735 | blocked on R-1 |
| `coremark_matrix` | 14343 | blocked on R-1 |
| `beebs_crc32` | 1703161001 | blocked on R-1 |
| `beebs_insertsort` | 271779359 | blocked on R-1 |
| `beebs_janne` | 484656629 | blocked on R-6 |
| `rv8_sha512` | 1390718314 | blocked on R-7 → R-1 |
| `rv8_sha512s` | 2842840124 | blocked on R-1 |
| `beebs_ns` | 1184999093 | blocked on R-9 |
| `rawhazard5/6/7` | 48879 each | R-1 discriminator |

## Pre-registered

1. **The anchors reproduce.** `ctrsanity` reads **1.000×** (identical code both sides); `beebs_prime`
   **1.054×**; `beebs_aha_mont64` **1.023×** at instret **256,699**. Instruction counts are
   deterministic and must land at or within a few counts of the published values; cycles may move a
   little on layout, which is documented (2026-07-26: four added instructions flipped a rung, and
   mont64 re-measured 290,071 against 289,869, a 0.07 % difference).
2. **The four R-1 rungs RETURN their oracles.** R-1 is recorded GONE, and these four are named in its
   impact line. This is the load-bearing prediction.
3. **`rawhazard5/6/7` return 48879.** They are the discriminator: if a rung hangs, these say whether
   R-1 came back or the cause is something else.
4. **Deliberately unpredicted: `beebs_janne`, `beebs_ns`, `rv8_sha512`, `rv8_sha512s`.** Their entries
   are R-6, R-9 and R-7, not R-1, and the archive is explicit that *"two of the eight are not explained
   by R-1"* and that writing "all remaining failures are the register-indexed-load defect" was
   falsified the next day. Whether they pass is the open part of this run.

## Refutation and VOID criteria

- **If any anchor's instruction count differs from its published value by more than 1 %, the vehicle is
  not comparable with the published table.** No new row may be added to §2, and the run is reported as
  a vehicle finding rather than a measurement. This is the condition that makes prediction 1 a gate and
  not a decoration.
- **A boot whose control fails is VOID** and carries no verdict about anything.
- **A hang is a RESULT**, not a missing datum: it is reported as a platform limitation with its issue
  number. Every rung returns a verdict because the runner power-cycles per rung, so no rung's failure
  costs another rung's reading.
- A returned value that is **not** the rung's oracle is a wrong-answer result and is reported as such —
  never counted as a measurement.

## Bitstream

`caplifive_m1_054cea69b.bit`, resident, not reflashed. `FPGA_ALLOW_FLASH` is unset;
`run_ladder_perf_fpga.py` is the only runner that can flash and it must not here. Timing does not
close on this build (WNS −8.307), so no wall-clock or MHz-normalised figure may be derived.

## `run-board-ladder.sh` self-deadlocks against the driver's own board lock (found 2026-09-22)

The one-line launcher cannot start a run. `run-board-ladder.sh:44` takes a non-blocking `flock` on
`/tmp/capstone/.board.lock`, then execs `run_ladder_perf_fpga.py`, whose `fpga_console.connect()`
calls `_acquire_board_lock()` on **the same path** and fails:

```
BlockingIOError: [Errno 11] Resource temporarily unavailable
RuntimeError: another board session holds /tmp/capstone/.board.lock
       holder: pid=<the launcher itself> ... rungs=ctrsanity beebs_prime ...
```

The wrapper reports its own shell as the conflicting holder. `flock` is per open file description, so
the child's separate `open()` contends with the parent's lock rather than inheriting it. The launcher
predates the locking that was later added inside `fpga_console.py`, and the two have not been run
together since.

Consequence for this run: the driver is invoked directly with the launcher's environment
(`FPGA_URL`, `FPGA_FW`, `LADDER_ONE_BOOT`, `LADDER_DISTINCT_VA`, `DOMAIN_GLUE`, `LADDER_RUNGS`), so
exactly one lock is taken — the driver's, which is the one that actually serialises the board. The
launcher's firmware-size, monitor-label and payload-freshness checks were run and passed first, so
nothing it guards is skipped.

**Not fixed here.** A launcher fix is a separate change and must not ride along with a measurement
pre-registration.
