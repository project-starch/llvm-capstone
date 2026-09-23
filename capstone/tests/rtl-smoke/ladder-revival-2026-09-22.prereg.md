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

---

# Phase 2 addendum — a control the baseline half can actually measure (2026-09-22, before the run)

Phase 1 produced no overhead rows because `ctrsanity` failed as a control. The runner's own evidence
column said why: **`clean = 1/15` in every one of the three counter-probe runs**, with the minimum
instret varying by 6,415 between them. At ~500,000 instructions it never completes one uninterrupted
pass in Linux userspace, so its floor is never reached. `beebs_janne` fails the other way — clean at
15/15 but only ~200 instructions, so bracket scaffolding is ~6 % and it reads 0.938×.

Two changes, both made before this run:

1. **`run_ladder_base_fpga.py` now reports the floor, not pass 2.** The minimum-instret warm pass was
   already computed and logged as `BEST`; the summary table printed pass 2 beside it. Pass 2 is one
   sample of a distribution whose spread reaches 350 % on identical code. The table now prints
   `best_cyc`/`best_ins` plus **`clean` = passes tied at that minimum**, which is the evidence the
   floor was reached, and flags `clean < 2` as **floor NOT reached**. Verified by replaying the
   2026-09-22 capture through the same logic: it reproduces every `BEST` line and flags `ctrsanity`.
2. **New rung `ctrsanitys`** — `CTRSANITY_N = 1000`, ~1/100th of `ctrsanity`, same kernel, same
   `-O1`, identical code on both halves. It targets the window the phase-1 data brackets: 7,272
   instructions read 14/15 clean, 25,666 read 2/15.

## Pre-registered, before the boot

1. **`ctrsanitys` reads `clean` ≥ 10/15 on the baseline half.** This is the whole point of the rung;
   if it does not, the window hypothesis is wrong and the baseline half cannot measure a control at
   any length, which is a bigger finding than the overhead table.
2. **`ctrsanitys` reads an instruction ratio within 1 % of 1.000.** It is identical code on both
   halves. **This is the gate: if it does not, no overhead row is published**, exactly as in phase 1.
3. **Deliberately unpredicted: the cycle ratio.** `ctrsanity` published at 1.000× cycles on the July
   bitstream, but that bitstream's CPI has since moved 16.7 % on this very kernel, so predicting the
   cycle ratio here would be predicting the new silicon rather than testing it.
4. **`ctrsanity` itself is run again as the NEGATIVE control** — it should once more read
   `clean = 1/15` or similar. A control rung that is supposed to fail and doesn't would mean phase 1's
   diagnosis was wrong.

## Refutation and VOID

- `clean < 10/15` on `ctrsanitys` refutes prediction 1 and the run publishes no ratios.
- An instruction ratio outside 1 % on `ctrsanitys` blocks every row, as in phase 1.
- Both halves are built from the same `ladder-rungs.spec` at `-O1` with `LADDER_OPT` unset, and the
  baseline half's own `capability-instructions=0` check must pass.
- Oracles frozen: `ctrsanitys = 3688591409`, QEMU parity exit 0 before any board time.

---

# Phase 3 addendum — re-measure the EIGHT PUBLISHED rows on this bitstream (2026-09-23, before the run)

Phase 2 produced ten rungs of overhead against a clean bare-metal baseline but could not merge them
into §2, because that table's control reads **1.000×** and this bitstream's reads **1.167×**. One
consistent table settles it. All eight published baselines were already captured by the 2026-09-22
bare-metal sweep at `15/15, spread 0`, and seven of them reproduce the published denominators exactly,
so only the capability half is missing for five rungs: `beebs_cnt`, `beebs_bs`, `beebs_recursion`,
`beebs_cover`, `rv8_primes`.

## Ordering, and the one rung expected to fail

`LADDER_ONE_BOOT=1` runs the sweep in one boot, so a wedge costs every rung after it. **`rv8_primes`
goes last**, alone in that position, because the published table's own footnote says it *"HANGS at
−O1 on this silicon and is measurable only at −O0"* and the spec now carries it at −O1. QEMU parity
cannot discriminate here — it passed at −O1 — because the documented failure is a silicon one.

## Pre-registered

1. **`beebs_cnt`, `beebs_bs`, `beebs_recursion`, `beebs_cover` return their oracles.** All four have a
   clean baseline and a published capability row, so a failure would be a regression against a
   measurement that already exists.
2. **Their cycle ratios move UP relative to the published table**, in the same direction and rough
   proportion as the control's 1.000 → 1.167. That is the hypothesis this run exists to test: if the
   control's shift is a uniform property of capability-mode execution on this bitstream, every row
   should carry it. **If the rows move by materially different amounts, the control's 16.7 % is not a
   uniform penalty and no single correction can reconcile the two tables.**
3. **Deliberately unpredicted: `rv8_primes` at −O1.** Either it hangs, confirming the footnote holds on
   this bitstream, or it returns, which retires a documented platform limitation. Both are results.
4. **Instruction ratios stay within a few percent of the published ones**, since both halves are built
   from one spec at one `-O`. A large instruction-ratio move would mean the comparison is measuring
   codegen drift rather than silicon.

## Refutation and VOID

- A rung returning a value that is **not** its oracle is a wrong-answer result, reported as such and
  never counted as a measurement.
- If `rv8_primes` hangs, every rung after it is lost — there are none, by construction.
- Baselines are **not** re-run: they are the 2026-09-22 bare-metal figures, `15/15, spread 0`, which
  reproduce the published denominators to the digit. Re-using them is what makes this a one-boot job.
- Oracles frozen and QEMU parity exit 0 before any board time: `beebs_cnt` 2356896837, `beebs_bs`
  887447230, `beebs_recursion` 1579141629, `beebs_cover` 1993178309, `rv8_primes` 99991.

---

# Phase 4 addendum — the length series on the control kernel (2026-09-23, before the run)

Phase 3 **refuted** phase 3's own prediction 2. The eight published rows did not shift uniformly with
the control: four are stable within ±1.3 %, two moved because their codegen improved (`beebs_cnt`
instruction ratio 1.319 → 1.197, `beebs_bs` 1.058 → 0.992), one is −O-confounded, and **`ctrsanity`
alone moved +16.7 %** at an instruction ratio of 1.000 both times. So the control's shift is a
property of that kernel, not of capability-mode execution on this bitstream.

Within that kernel the effect tracks length: `ctrsanitys` (5,021 instructions) reads **1.045×**,
`ctrsanity` (500,022) reads **1.167×**. `ctrsanity4` is the same kernel again at 2,000,022, its
baseline is already measured at `15/15, spread 0`, and separating this is the reason the pair exists —
its own header: *"Two lengths separate a PROPORTIONAL counter effect (ratio unchanged as work grows)
from a FIXED one (ratio moves toward 1.0)."*

**All three lengths run in ONE boot** so that boot-to-boot variation cannot be mistaken for the effect.

## Pre-registered

1. **The two shorter rungs reproduce their previous readings** — `ctrsanitys` 1.045×, `ctrsanity`
   1.167×, from separate boots. If they do not, the effect is boot-dependent and neither earlier
   number means what it was taken to mean.
2. **`ctrsanity4` exceeds 1.167×** if the mechanism keeps accumulating with run length. **Roughly
   1.167×** if it has saturated by 500k instructions. **Below 1.167×** would mean the ratio peaks and
   falls, which no fixed-or-proportional account predicts and which would refute the framing rather
   than the number.
3. **Instruction ratio stays 1.000** on all three. It has been 1.000 on this kernel at every length
   and in both vintages; a move would mean the comparison stopped being about cycles.
4. **Baseline CPI stays flat at 1.200.** Measured across a 400× length range yesterday
   (1.203 / 1.200 / 1.200), so the bare-metal denominator contributes nothing to the rise. Not re-run
   here — it is quoted from the 2026-09-22 sweep.

## Refutation and VOID

- Any rung returning a value that is not its oracle is a wrong-answer result, never a measurement.
- If prediction 1 fails the run is VOID as a length series, because the points would not be comparable.
- This characterises the effect; it does not explain it. No mechanism is offered in advance, and none
  may be written afterwards from these three points alone.

---

# Phase 5 addendum — `rv8_primes` at −O0, the measurement phase 3 assumed (2026-09-23, before the run)

## What this closes

Phase 3 measured `rv8_primes` at **−O1** (1.005×, instruction ratio 1.000) and §2 then called the
published **1.263× at −O0** "largely an optimisation-level artefact". **That sentence outruns its
evidence.** The two numbers differ in *two* variables, not one:

- the **−O level** (−O0 published, −O1 measured), and
- the **bitstream and compiler** (July 2026 vs `caplifive_m1_054cea69b`).

No −O0 measurement exists on the current silicon, so "artefact of the optimisation level" is an
attribution that was never tested. Every other row in the re-measured table is a like-for-like pair;
this one is not, and it is the only row where the shift was explained by a variable that was not
isolated.

## Design — the row controls itself

Both halves are rebuilt with `LADDER_OPT=-O0`, which the spec supports precisely so the two halves
cannot drift (`build-ladder-fpga.sh:54`, `build-ladder-base-fpga.sh:61`). The published `rv8_primes`
row **is** an −O0 row, so reproducing it is the control: an −O0 pair taken today is directly
comparable to the −O0 pair published in July, with the −O level held fixed and only the
bitstream/compiler varying. That is the variable phase 3 could not separate.

`ctrsanity` rides along as an independent known-good control (oracle correctness and `clean`/spread
only — its ratio at −O0 is NOT comparable to its published −O1 1.000×, and will not be quoted).

Rungs, in order: `ctrsanity rv8_primes`. Both expected to return.

## Pre-registered, before the boot

Published −O0 reference: capability **17,283,292 / 8,773,753**, baseline **13,679,903 / 7,764,899**,
cycles **1.263×**, instructions **1.130×**, CPI 1.118.

1. **`rv8_primes` returns oracle 99991 at −O0.** C-3 has read `GONE` since 2026-09-05 and the rung
   already returned its oracle at −O1 (phase 3) and −O2 (the C-3 closure). A non-return here refutes
   the C-3 closure rather than anything about −O levels, and is reported as such.
2. **The INSTRUCTION ratio is the discriminator.** *(Corrected before the run: an earlier draft of
   this addendum said it "is decided at the desk, before the board". It is not — instret is a
   DYNAMIC count, and the parity runner `run-ladder-perf-qemu.sh` is a correctness gate that reports
   no counters at all. The ratio is read from the board run's own instret column. The discriminating
   LOGIC is unaffected: instruction counts are emitted by the compiler and merely counted by the
   silicon, so the ratio still separates a compiler change from a silicon one — only the claim about
   where it gets read was wrong.)*
   - **reproduces ~1.130** → the −O0 codegen is materially unchanged since July, and any cycle-ratio
     movement is the silicon's;
   - **has collapsed toward 1.000** → the capability compiler improved at −O0 too, and the cycle
     shift is the compiler's, exactly as for `beebs_cnt`/`beebs_bs`.
3. **The CYCLE ratio reproduces 1.263× within ±3 %.** If it does, the published row stands on current
   silicon, and "−O1 removes this overhead" becomes a clean, isolated optimisation finding — the
   strongest available version of §2's claim, now earned.
4. **REFUTATION, named:** if the −O0 cycle ratio comes back at ~1.00×, then **the published 1.263× does
   not reproduce on this bitstream at its own −O level**, the "optimisation-level artefact" attribution
   in §2 is **false**, and the cause lies in the bitstream or the compiler. That sentence is then
   struck, not softened.
5. **`ctrsanity` returns oracle 43260934** with `15/15` passes tied at min instret, spread 0, on both
   halves. A `clean` below 15/15 on the bare-metal baseline contradicts I-2's closure and VOIDs the run.

## Refutation and VOID

- **VOID** if `ctrsanity` fails its oracle on either half, or if the baseline half's `clean` is not
  15/15 at spread 0 — the denominators are then not floors and no ratio may be read.
- **VOID** if the result file's mtime predates the run (the phase-2 staleness trap: a failed run
  writes nothing and a stale artifact copies clean). Delete before, check mtime after.
- **Not a VOID, a result:** `rv8_primes` failing to return at −O0. That is reported against C-3.
- **No mechanism will be offered for whatever the cycle ratio does.** As in phase 4, two points
  bound a difference; they do not explain one.

## Bitstream

`caplifive_m1_054cea69b.bit` (WNS −8.307), resident. `FPGA_ALLOW_FLASH` stays unset.
