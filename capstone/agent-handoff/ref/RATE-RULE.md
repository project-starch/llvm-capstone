# The rate rule — why "X wedges" is not a result on this platform

**Rescued 2026-08-18 from `SILICON-BLOCKER.md:5170-5198`, where it was the single most
consequential paragraph in the handoff set and sat 5000 lines into a document nobody reads to the
end.** That document is superseded; this rule is not. It is the reason most of it is unsafe to
build on.

## The rule

**A single-sample wedge means nothing.** Most wedges on record are single samples *by
construction*, because a wedge ends the board session — so "X wedges" is, on its own, consistent
with pure background.

**The measured unit is a RATE, with n reported.** "X failed" is not a result. "X failed k of n"
is.

**Every A-passes / B-fails pair is suspect** unless both arms were sampled repeatedly. Historical
pairs that were single-sample per image — `wd66`/`wd85`, `wd77`/`wd78`, guarded vs unguarded,
stage 85 vs 86, the ballast ladder — do not support the conclusions drawn from them.

**A control that PASSES proves the board booted and the image loaded. It does NOT make the
following wedge meaningful.**

## What it explains

The "unexplained build-to-build sensitivity" that runs through the entire silicon investigation —
identical logic behaving differently across binaries, with and without padding, guarded vs
unguarded — is, at least in part, this background rate sampled once per image. The mechanisms
proposed and then retracted (walk count, first-walk anomaly, accumulator loss, cap-init threshold,
layout, instruction placement) were all inferred from exactly that kind of single-sample
comparison.

## Measured, so the rule has numbers behind it

`XU` at hash `f1214600d0dac351`, one bitstream, no reflash between runs:

* wedged at 03:42, **passed at 03:50** — same hash, same physical base `0x84400000`, same position
  in the boot, eight minutes apart;
* passed 4 of 4 at 16:19;
* passed, passed, **wedged** at ~20:00 — the flip occurring *inside a single boot*.

**k = 1 in n = 7** on 2026-08-18. The wedge rate is not a property of the image, the bitstream, the
physical placement, or the position in the boot. All four were tested and none correlates.

## Practical consequence

Reps are cheap, but there ARE ceilings and the earlier "no ceiling" claim here was wrong.
It came from 10 identical *rungs* passing in one boot — rungs consume ~1 region each, so that
negative could not fire. Two real per-boot ceilings exist:

* **the region table** — `CAPSTONE_MAX_REGION_N`, was 32, raised to 96 on 2026-08-18 and
  demonstrated at 12 domains / `rgid` 58 with zero overflows;
* **`SPLB:0000E010` = `CAPSTONE_ERR_SPLIT_EXACT`**, the `split_out_cap` exact-fit spin, still
  unfixed. It is pool-state dependent, not a fixed count: with the same `S7T` + `XU` sequence it
  stopped one boot after 2 `XU` reps and another after 6.

So a boot yields roughly 6-12 reps, not unlimited ones, and a boot cut short by `SPLB` is
**right-censored** — it carries "no wedge in N reps", never a verdict about rep N+1. There is
still no excuse for an n of 1.

Accumulate k/n opportunistically by appending reps to boots that are happening anyway, and
classify failures before counting them — an S-07 wedge dies *after* `SQ: G/enter`; a resource or
entry-stall failure dies before it, and must never be counted as a defect.

## Running tally on `caplifive_s07debug_18august.bit` (`capstone-ariane` `6882b265f`)

`XU`, hash `f1214600d0dac351`, one domain repeated after an `S7T` control, uniform geometry:

| boot | reps | wedges | reps until first wedge |
|---|---|---|---|
| 1 | 3 | 1 | 3 |
| 2 | 2 | 1 | 2 |
| 4 | 6 | 0 | none — censored at 6 (`SPLB` stopped the boot) |
| 5 | 1 | 1 | 1 |
| 6 | 6 | 1 | 6 |

**k = 4 of n = 18.** Boot 3 was VOID. `NO-ENTRY` and `SPLB` stops are excluded from both k and n
by `tests/rtl-smoke/s07-rate.py`.

The v4 question — per-BOOT state (bimodal) vs per-RUN randomness (geometric) — is **not yet
decided**. Five boots giving first-wedge positions **1, 2, 3, 6 and one censored at >6** now lean
clearly toward **per-RUN randomness**. The MLE is p = 4/18 = 0.22 per rep, and a geometric at
that p predicts first-wedge positions scattered across exactly this range with P(>6) = 0.22 --
one censored boot in five is what it expects. A per-BOOT model predicts the opposite shape:
clustering into boots that wedge almost immediately and boots that survive the full ladder. We
see neither cluster; we see a spread.

**Not yet conclusive, and the honest reason is n.** Five first-wedge observations cannot
separate a geometric from a mild mixture, and the ladder is capped near 6-7 reps by SPLB, which
censors exactly the tail where the two models differ most. But the direction is now consistent
enough that the follow-on work should assume **per-run** -- hunt the faulting site, not the
environment -- rather than spend boots on power-cycle dwell or thermal variation.

## THE CONTROL HAS NO n EITHER — and that is the next thing to fix

Tallied across every boot on this bitstream:

| domain | k | n | note |
|---|---|---|---|
| `XU` (the arm under test) | 4 | 20 | p̂ = 0.20 per rep |
| `S7T` (the CONTROL) | 0 | **6** | one rep per boot, six boots |

**`S7T` is not established as immune, and the pair it forms with `XU` is not supported.** Under
the null that `S7T` behaves exactly like `XU`, P(0 wedges in 6) = 0.80^6 = **0.26**. A quarter of
the time we would see this even if the two domains were identical in every way that matters.

This is the rule at the top of this document applied one level up, to the instrument rather than
the subject. Every boot on record has run `S7T` **once**, as a control, and then read "the control
passed, the arm wedged" as though it were a comparison. It is a comparison between n = 6 and
n = 20, one of which has never had the chance to fail.

**The next boots run `S7T` REPEATED**, to give it a real n:

* 0 in ~20 reps puts P at 0.012 under the null. Only then is `S7T` vs `XU` a genuine matched pair
  worth diffing, and the diff becomes the localisation.
* A comparable rate means **`S7T` is not a control at all**, and every boot that read "control
  passed, therefore the boot is good and the wedge belongs to the arm" was reading nothing. That
  would be the more consequential result of the two.

**No separate control domain in those boots, deliberately.** The control exists to separate "this
image failed" from "the board or firmware failed", and `SQ: G/enter` already makes exactly that
distinction per rep — a domain that entered and wedged is a result, one that never entered is
excluded from both k and n by `s07-rate.py`. Adding a small control domain would instead perturb
the carve geometry that the `SPLB` exact-fit ceiling is sensitive to, trading a discrimination we
already have for a new source of lost reps.

## The rate is over a HOMOGENEOUS population — checked, not assumed

A rate computed over a mixture of failure mechanisms is not a rate for either of them. There are
at least two distinct ways to hang on this design: an S-07 capability fault (mcause 25), and a
rev-node pool stall, which `capstone_rev_node.anvil:99-108` produces as a **blocked `recv` with no
trap at all** — deliberately, so exhaustion is a visible stall rather than silent id reuse.

Those are distinguishable on data already collected, from switch 255, `{seen, mcause[6:0]}`:

| boot | outcome | sw=255 | seen | mcause |
|---|---|---|---|---|
| 1 | S07-WEDGE | 0x99 | 1 | **25** |
| 2 | S07-WEDGE | 0x99 | 1 | **25** |
| 5 | S07-WEDGE | 0x99 | 1 | **25** |
| 6 | S07-WEDGE | 0x99 | 1 | **25** |
| 3 | SPLB (infra) | 0x89 | 1 | 9 |
| 4 | SPLB (infra) | 0x89 | 1 | 9 |

**All four counted wedges trapped with mcause 25**, each corroborated by an independent latched
trap at a different address. A no-trap stall cannot present as a trap, so **rev-node exhaustion is
excluded as the mechanism for every wedge in the rate.** k=4/n=20 is a rate for one thing.

### But the SPLB ceiling is a different animal, and it is not excluded

The two boots with no capability trap are exactly the two that ended in `SPLB:0000E010` =
`CAPSTONE_ERR_SPLIT_EXACT`. Cause 9 is an S-mode ecall — ordinary traffic latched earlier, saying
nothing about the domain. Those stops are already excluded from k and n, and should stay excluded.

The ceiling is worth understanding on its own account, because it censors every ladder at 6-7 reps
— precisely the tail where a **rising** hazard (cumulative allocation across reps) separates from
a **constant** one, and the only region the ladder never observes. `SPLB` is split-related and
allocation-related and fires at a cumulative point, which fits a rising hazard.

What does not fit: `SPLB` is the **monitor's own error code**, so the monitor is running and
reporting rather than stalled, and the hardware SPLIT returned. Pool pressure short of exhaustion
would reach the monitor as an exact-fit failure; exhaustion itself would not reach it at all. And
boot 4's wedge readout gave `head = 365` (10 bits of 16, so treat with care — a truncated head has
been misread as exhaustion here before) with `overflow = 0`, and `serving_idx = 0x00000000`, which
is either not advancing or not wired as we read it.

**Next, and it needs no RTL:** read the head and `serving_idx` before rep 1 and after *every* rep
rather than only at a wedge. That gives consumption per rep directly, which answers whether the
pool can reach exhaustion in 6-7 reps at all, and whether `SPLB` tracks allocation.

## PRE-DECLARED ANALYSIS for the S7T-vs-XU comparison

**Written and committed BEFORE the first interleaved boot runs.** That is the entire point: with
two defensible analyses pulling opposite ways, whichever is chosen after the data arrives will be
the more favourable one, and nobody — including whoever chooses it — will be able to tell whether
that was the reason. So the choice is made here, in advance.

### The design

`S7T` and `XU` **interleaved in the same boot**, alternating, with the **leading arm counterbalanced
across boots**: boot 8 leads `S7T`, boot 9 leads `XU`, boot 10 leads `S7T`, boot 11 leads `XU`.

Counterbalancing is not fussiness. Under strict alternation with a fixed leader, `S7T` occupies
positions 1,3,5,7 (mean 4) and `XU` occupies 2,4,6,8 (mean 5), so the arms are compared at
systematically different **depths** in the ladder — and depth is precisely the variable under
investigation. If the hazard rises with position, `XU` looks worse by construction and the effect
is confounded with the thing being measured. Four boots makes the balance exact; if only three are
run, the residual imbalance is recorded, not ignored.

Interleaving rather than dedicating boots to each arm removes the boot-to-boot confound, which is
the one this project has repeatedly been bitten by — the same hash behaving differently across
boots is what this whole exercise exists to characterise. A boot that dies early then costs both
arms equally instead of biasing one.

**Why the never-interleave warning does not apply here:** it is about heterogeneous SIZES
perturbing the carve geometry `SPLB` is sensitive to. The case that produced it was a ~10 KB
domain against a ~1.5 MB one. `S7T` (1548496 B) and `XU` (1551888 B) are **0.219% apart**. The
warning stands in full for heterogeneous sizes, and applies again if anyone later interleaves a
small probe domain with these.

### PRIMARY — paired, valid, and underpowered

Fisher's exact, one-sided, `S7T` against `XU` **within the interleaved boots only**. Computed, not
asserted:

| table | p |
|---|---|
| 0/12 vs 3/12 | 0.109 |
| 0/12 vs 4/12 | 0.047 |
| 0/20 vs 4/20 | 0.053 |
| 0/26 vs 6/26 | 0.011 |

So the realistic outcome at n=12 per arm — 0 against ~3 — **does not reach 0.05**, and n=20 per arm
is borderline. **This is stated in advance so that a null result is read as low power rather than
as evidence of no difference.** Target n≈20 per arm, which is 4 boots at ~5 reps per arm per boot.

### SECONDARY — pooled against the historical rate, higher-powered and confounded

`S7T`'s count against the already-collected `XU` rate p̂ = 0.22 as a known value:

| | P(0 wedges) |
|---|---|
| n = 12 | 0.051 |
| n = 20 | 0.007 |
| n = 26 | 0.002 |

Higher-powered, but it reintroduces exactly the boot-to-boot confound the pairing exists to
remove. It is the **secondary** analysis and does not get promoted if the primary disappoints.

**If the two agree, the conclusion is strong despite each being individually weak. If they
disagree, that disagreement IS the finding** — it says boot-to-boot variation is real and large,
which is worth more than either analysis alone.

### The outcome that needs no statistics

**If `S7T` wedges even once with mcause 25 at a comparable position, the discriminator is dead on
the spot.** A single event in the control arm refutes rather than estimates, and no power
calculation is required or relevant. Check for that before running any test.

## RETRACTION: `S7T` vs `XU` cannot localise anything, at any n

**Retracting the claim, made earlier on this page and in a report, that a clean `S7T` at n~20
would make `S7T` vs `XU` "a genuine matched pair worth diffing" and that "the diff is the
localisation".** It is not a matched pair and the diff localises nothing. Source, not inference --
`benchmarks/sqlite/sqlite_capstone_domain.c:6737-6743`:

    #ifdef CAPSTONE_S07_CURSOR_SELFTEST
        /* Runs INSTEAD of the workload: this arm exists to prove the instrument, and it must be
           an expected-to-RETURN arm so it can be ordered before the one that may wedge. */
        *res = s07c_selftest_result();
        return;
    #endif
        unsigned long rv_ = (unsigned long)(unsigned)run_sqlite();

`S7T` is built with `SQLITE_S07_CURSOR_SELFTEST=1` and **returns before `run_sqlite()` is ever
called.** It plants two known integers into a 16-byte slot, reads them back, and returns. It does
not execute the workload at all.

So "the control passed and the arm wedged" is not an underpowered comparison -- that was the
previous diagnosis and it was too generous. It is a comparison between a program that runs SQLite
and a program that does not. A clean `S7T` is the expected result of not running the code under
test, at any n, and no number of reps converts it into evidence about the workload.

**The power calculation above was computing the power of a comparison that cannot localise
regardless of the answer.** The arithmetic is still correct and the pre-declaration still stands
for what it does measure; it simply measures less than was claimed for it.

### What the comparison IS still good for, and it is not nothing

* **Validating the boot.** That is what a control is for, and `S7T` does it: it proves the board
  booted, the firmware loaded, the domain was created and entered, and the instrument fires. Every
  past boot keeps that much.
* **One genuinely informative outcome remains.** If `S7T` ever wedges with mcause 25, then a
  domain that never runs the workload has wedged -- which points at the monitor, the entry path or
  the carve, NOT at SQLite. That would be a large finding, and it is why the interleaved boots are
  worth finishing rather than aborting.

### What an actual matched pair requires

Two builds that **both run the workload** and differ in exactly one thing -- a single fixup or
codegen knob toggled (`build-sqlite-silicon.sh` carries several, e.g.
`SQLITE_LDC_HIGH_HALF_FIXUP`). That is the pair whose diff localises, and it is what the next bake
should produce. `S7T` was never a candidate for it.

**How this got missed:** `S7T` has been called "the control" in every transcript and doc for long
enough that nobody re-read what it compiles to. The name asserted a relationship the code does not
have, and the reasoning was done on the name -- including by me, twice today, while I was in the
middle of correcting two other instrument-label errors of exactly the same shape.

## THE HALTED READ WORKS, AND IT REFUTES THE 0x9c CASE

Boot 8, early halt control on a healthy core, before any domain ran:

    running 208 (pre-run baseline)  0x9c  10011100
    halted  208 (same boot)         0x90  10010000

| field | running | halted |
|---|---|---|
| ldc0_valid | 1 | 1 |
| src | 0 | 0 |
| stc_valid | 1 | 1 |
| **stc_ctag** | **1** | **0** |
| **gran_match** | **1** | **0** |
| clobbered | 0 | 0 |

`0x90 & 0x9c == 0x90` — the halted read is a **strict subset**, exactly as the stretcher predicts
(contamination can only add bits). And the two bits present *only* in the running read are
precisely `stc_ctag` and `gran_match` — **the two that made it decode as "(b) GENUINE TAG LOSS"**.

### Two conclusions, both load-bearing

**1. The halted read WORKS, and it is now the protocol.** The claim that halted reads were
structurally impossible is withdrawn: it rested on a `latest()` fallback (since removed) and on
n=1 in a total-wedge state. The mechanism is the RTL lane's: `clk` is the free-running MMCM output
(`ariane_xilinx.sv:1209`) and the stretcher counts down on it (`:961`) regardless of the hart, so
with the mux input static every undriven bit decays in ~21 ms while the driven bits reload.
Halting is the one configuration where contamination *clears*. The driver now halts around every
per-domain mux read (`HALT_MUX_READS`, default on) and forces the settled value to be emitted by
walking to aperture 0 and back, since the board pushes `led_state` on change only.

**2. The boot-time "S-07 signature" is REFUTED.** The reproducible pre-run `0x9c` — filed as a
candidate cheap repro, an untagged LDC on the same granule as the most recent tagged STC — was
pulse-stretcher contamination. Caveat 5 in the spec was right and is the reason this was filed as
a case to test rather than reported as a finding: **bit-identical across boots did not clear it,
because the switch walk is deterministic and so is the contamination it produces.**

### Corollary: every running non-zero mux reading ever taken is void

Not merely suspect — void, and now demonstrably so. That includes the `rev-node head = 65535`
readings taken running in boot 8 (all-ones is the saturated case) and the `0xfe`/`0xbe`
displacement bytes. Zeros still stand, running or halted, because contamination cannot manufacture
a zero. Re-take anything non-zero that mattered, halted.

## The per-domain halted read cannot work as written — diagnosis, not defeat

`monitor halt` between domains times out on **every** domain. Boot 13, the decisive three lines:

    emit gdb_input <- {'text': 'monitor halt\n'}
    gdb_output: {'data': 'monitor halt\r\n'}
    [s07] halt before mux read FAILED (ActionTimeout)

The command is **echoed** and no prompt ever returns. After `continue`, **GDB is not at a
prompt** — it is blocked until the target stops — so a command sent there sits in its input
buffer and `gdb_cmd` waits forever for a `(gdb)` that cannot come.

Ruled out along the way: session churn (holding one session for the whole run changed nothing),
the wedge state, and the board.

**The fix, for whoever picks this up:** interrupting a RUNNING target needs an **interrupt**, not
a command — send `\x03`, wait for the stop, then issue `monitor halt` / the reads, then
`continue`. That is a different mechanism from anything in the driver today and deserves to be
written and negative-tested deliberately.

**Why the EARLY control works and this does not**, which is the whole difference and was visible
for four boots: the early control issues its halt while gdb is **at a prompt**, immediately after
`gdb_start()`. Same helper, same apertures, same board.

`HALT_MUX_READS` now defaults **off**. The timeout costs ~30 s per domain and every reading it
produces is a running read that the closed encoding then correctly rejects, so leaving it on buys
nothing and slows every boot.

### What is known-good and stays

* **the rate**, from UART markers — untouched by any of the mux trouble;
* **the early halt control**, which is where the `0x90` vs `0x9c` subset proof came from;
* **the closed-encoding rejection**, which caught every bad byte today without being asked.

### And a failure mode worth naming separately

The four instrument-label errors were checks that could not fire. This one is the opposite: **a
check that fired correctly into a log nobody re-read.** `halt before mux read FAILED — readings
below are RUNNING reads and every non-zero one is void` printed above every affected reading, and
the `0x7c` those reads returned was investigated as a hardware mystery for two boots. Reporting a
failure is not the same as the failure being noticed.

## SUSPECTED: the early halt control may be perturbing the rate it is measured beside

Splitting every `XU` rep on this bitstream by whether the **early halt control** was present:

| era | boots | k | n | p-hat |
|---|---|---|---|---|
| before | 1-7 | 4 | 27 | 0.148 |
| after | 8-14 | 5 | 9 | **0.556** |

Fisher exact, one-sided: **p = 0.026**. Reps-to-first-wedge tell the same story — `3, 2, >6, 1,
6, >7` before, against `2, 1, 3, -, -, 1, 1` after.

**This is a HYPOTHESIS GENERATED FROM THE DATA, NOT A TEST, and it must not be reported as one.**
The comparison was chosen after seeing the numbers, which is precisely the selection the
pre-declared analysis on this page exists to prevent. Holding the same standard here:

* **The eras differ in more than one thing.** The early halt control arrived at boot 8, but so did
  the interleaved `S7T`/`XU` ladder, and boots 11-13 additionally issued per-domain `monitor halt`
  commands that timed out. "Early halt control" is confounded with at least two other changes.
* **n = 9 in the post era**, most of it from boots that wedged on rep 1 and therefore contributed
  a single rep each.
* A post-hoc p of 0.026 across many possible splits of the data is worth roughly what it looks
  like: a reason to test, not a result.

### Why it matters enough to spend boots on

If halting the core before the workload runs changes the wedge rate, then **the only working
halted read contaminates the measurement it sits beside**, and the k/n on this page is pooled
across two regimes. That would also mean every "instrument" conclusion today was drawn on a
system the instrument was altering.

### The proper test, running now

**Identical ladder, one variable.** Boot 15 repeats boot 14's exact interleaved ladder with
`EARLY_HALT_CONTROL=0`; subsequent boots alternate the flag. Nothing else changes — same domains,
same order, same count. That is the matched pair the era comparison is not.

Until it resolves, **do not pool boots 1-7 with 8-14**, and treat the headline k/n as provisional.

## PRE-DECLARED: the halt-control A/B

**Committed with n = 1 boot per arm on the board, before the rest is collected.** The era split
that motivated this was chosen after seeing the data; this one is not, and the difference is the
whole point.

### Design

**One variable.** Identical interleaved ladder every boot — same domains, same order, same count —
alternating `EARLY_HALT_CONTROL=1` and `EARLY_HALT_CONTROL=0`. Nothing else may change while this
runs. Boot 14 = ON, boot 15 = OFF, boot 16 = ON, and so on.

### Analysis, fixed now

Fisher's exact, one-sided, testing **ON worse than OFF**, on pooled `XU` reps per arm.

Power, computed before collection, under the era split's own estimates (OFF 0.15, ON 0.55):

| n per arm | expected table | p |
|---|---|---|
| 8 | 4/8 vs 1/8 | 0.141 |
| 12 | 7/12 vs 2/12 | 0.045 |
| 16 | 9/16 vs 2/16 | 0.012 |
| 20 | 11/20 vs 3/20 | 0.009 |

**Target n = 16 per arm.** n = 8 is underpowered and a null there means nothing; n = 12 is the
threshold and only if the effect is as large as the era split suggested. Stated in advance so a
null is read as low power rather than as absence of an effect — the same trap this page already
recorded once today for the `S7T` comparison.

### Standing data, not to be pooled with it

Boot 14 (ON): k=1 n=1. Boot 15 (OFF): k=1 n=4, wedging at rep 4. Two boots decide nothing and are
recorded only so the arms' running totals are auditable.

### If the effect is real

The early halt control is withdrawn from any boot whose rate is being measured, and the k/n on
this page is re-derived from `EARLY_HALT_CONTROL=0` boots only. The halted read remains valid as
an *instrument* — the `0x90` vs `0x9c` subset proof stands, since it is a statement about the
readout path and not about the wedge rate — but it may not run beside a rate measurement.

### If it is not

Boots 1-7 and 8-14 can be pooled again, and the era difference is attributed to the interleaved
ladder or to chance, both of which are testable the same way.

## FIRST LOCALISATION: the two granules are EXACTLY ADJACENT, and reproducibly so

Boots 15 and 17, independent, different wedging-domain `DBAS`, **bit-identical addresses**:

    untagged-LDC granule  0xaedc0
    last-cap-STC granule  0xaedb0
    delta                 -16 bytes = exactly ONE granule

Read halted, at the wedge, through the granule apertures that were in the bitstream all along
(205/206/207/209). These are the first trustworthy non-zero mux readings of the campaign.

### Why the addresses are meaningful rather than arbitrary

Both wedging domains had 4 MB-aligned `DBAS` (`0x85400000`, `0x80800000`), so bits `[19:0]` of
`DBAS` are zero and the exposed `[19:4]` **is an offset within the domain**, not a physical
address that varies per boot. That is why two boots at different physical bases produced the same
number, and it is what makes the value mappable at all.

### What it does and does not say

* **`gran_match = 0` is CORRECT here** — the records are one granule apart, not the same granule.
  So these wedges carry **no tag-loss claim** from the correlation bit, and any earlier reading
  that showed `gran_match = 1` is further evidence of contamination rather than of a match.
* **Adjacency is the finding.** Two independently rolling records landing exactly 16 bytes apart,
  identically, on two separate boots, is not what independent rollers produce by chance. The
  shape it suggests — a capability stored at one granule and a load coming back untagged from the
  neighbouring one — is a concrete, testable mechanism, and the first this campaign has had.
* **n = 2.** Two samples. Both from `EARLY_HALT_CONTROL=0` boots (see below for why only those).

### The mapping is NOT yet done, deliberately

`0xaedc0` falls inside `.text`'s VA range (`0x10000` .. `0x14eb28`) *if* the loader places the
segment at `DBAS + VA`; it falls elsewhere if it places it at `DBAS + (VA - 0x10000)`. **Which one
is right depends on the loader's placement rule, and that has not been checked.** Guessing it is
exactly the class of error that cost four separate corrections today, so the number is recorded
raw and the mapping is the next step, from the loader source rather than from inference.

## The early halt control BREAKS the wedge readout entirely

Separate from the rate question, and unambiguous:

| boot | arm | `[wedge] gdb CSRs` lines | granule addresses |
|---|---|---|---|
| 14 | ON | **0** | none |
| 16 | ON | **0** | none |
| 15 | OFF | present | **0xaedc0 / 0xaedb0** |
| 17 | OFF | present | **0xaedc0 / 0xaedb0** |

The early control leaves its gdb session open, and the wedge block's own `gdb_start()` then
fails, taking the *entire* wedge instrumentation with it — CSRs, trap latch, granule addresses.
So the ON arm silently loses every wedge diagnostic.

**Consequence: run `EARLY_HALT_CONTROL=0` for any boot whose wedge is worth reading**, which is
all of them now that the granule addresses work. The early control's own value is already banked
(the `0x90` vs `0x9c` subset proof) and does not need re-running.

### The mapping is AMBIGUOUS, and the tidy answer is the wrong one

Resolving the granules to symbols gives, on the naive mapping:

    VA 0xbedb0  (last cap STC)   inside sqlite3VdbeCreate  (start 0xbed38, size 0x170, +0x78)
    VA 0xbedc0  (untagged LDC)   +0x88 into the same function

**That is almost certainly wrong, and it is being written down as wrong rather than as a
result.** A capability *store* into a function body is not a sensible operation. What the tidy
symbol actually demonstrates is how easily an ambiguous number lands somewhere plausible.

`[19:4]` is 16 bits — **a 1 MB window** — and the domain spans more than 1 MB, so every reading
has one candidate per alias. The loader maps `DBAS` to VA `0x10000`
(`libcapstone.c:138`, `entry_offset = e_entry - p_vaddr`, both `0x10000`), giving:

| alias | untagged LDC | last cap STC | region |
|---|---|---|---|
| 0 | `0x0bedc0` | `0x0bedb0` | `.text` — **CODE**, implausible |
| 1 | `0x1bedc0` | `0x1bedb0` | past the image end (`0x1605a8`) — bss/heap |
| 2 | `0x2bedc0` | `0x2bedb0` | bss/heap |

Alias 1 or 2 — the region SQLite actually allocates from — is the plausible reading, and a
capability stored to one heap granule with the neighbouring one reloading untagged is a coherent
mechanism. Alias 0 is not, despite being the one that resolves to a named function.

**The upper address bits are not exposed on this bitstream**, so this cannot be resolved by
reading harder. It is the one thing in this investigation that genuinely needs new RTL — and it is
a strict extension of an aperture set that already exists, not a new instrument. The RTL lane's
caveat when handing over the aperture list said exactly this ("a granule within a 1 MB window, not
a full address; do not report it as a physical address"), and it was nearly ignored one step
later.

**Do not cite `sqlite3VdbeCreate` in connection with S-07.** It is an artifact of a 1 MB
ambiguity resolved in the wrong direction, and it is exactly the shape — a confident, legal,
plausible answer — that this page has now recorded five times in one day.

### THE SITE IS DETERMINISTIC EVEN THOUGH THE WEDGE IS NOT

Three independent boots, every one bit-identical:

| boot | wedging domain DBAS | untagged LDC | last cap STC |
|---|---|---|---|
| 15 | `0x85400000` | `0xaedc0` | `0xaedb0` |
| 17 | `0x80800000` | `0xaedc0` | `0xaedb0` |
| 19 | `0x80800000` | `0xaedc0` | `0xaedb0` |

**This is the strongest constraint the campaign has produced, and it is a new kind of one.**
Everything measured so far said the wedge is *probabilistic* — k=9 of n=36, first-wedge position
scattered from 1 to >7, the same image passing and failing minutes apart. The site is not. Same
two granules, same adjacency, every time, across different physical bases.

So the mechanism is **one specific location that intermittently loses its tag**, not a random
granule somewhere in a 1.5 MB domain. That rules out a large family of explanations — anything
that would spray across addresses (broad cache corruption, random bit flips, general refill
erasure) — and it makes the defect a property of *what lives at that address*.

It also makes the 1 MB ambiguity worth removing rather than worked around: with a deterministic
site, one number identifies the object, and the object is the answer.

**Caveat kept explicit:** boots 17 and 19 share a `DBAS`, so the independent-base evidence is
boot 15 against the other two, not three ways. And all three are `EARLY_HALT_CONTROL=0` boots,
because the ON arm loses the wedge block entirely.

### Narrowing without hardware: the site is OUTSIDE the image and outside the heap

`sqlite_heap` is a static `.bss` array — `0x160e90`, size `0x40000`, so VA `0x160e90 .. 0x1a0e90`
(`llvm-nm -S` on `XU.dom`). Checking every 1 MB alias of the granule under both candidate
mappings (`DBAS -> VA 0x10000` and `DBAS -> VA 0`):

**No alias falls inside `sqlite_heap`, and none falls inside the loaded image** (`PT_LOAD` ends
`0x1605a8`; the last allocated section, `.capstone_gp_table`, ends `0x1a20b0`).

So the faulting granule is in neither the static image nor SQLite's own heap. What is left is
**monitor-carved territory** — the domain stack, or a region mapped outside the image. A
capability spilled to the stack and reloaded untagged is a coherent and classic shape for this
defect, and it would explain a deterministic site under a nondeterministic trigger far more
naturally than a heap object would.

**That is a candidate, not a conclusion.** It rests on the alias enumeration and on the
`DBAS`-to-VA mapping, and the stack has not been located. The addresses the domain already prints
(`SQ: libc=`, `SQ: self=`) are host-process addresses (`0x3f...`, `0x2a...`) and do not help.

### Resolving it costs a domain rebuild, not a reflash

Have a **non-wedging** rep print its own image end, heap extent and stack pointer once — the site
is deterministic, so the object is the same on a clean run as on a wedging one — and match the
alias that falls inside a known region. One rebuild, no synthesis, no reflash, and **the rate
campaign survives**: a new bitstream resets every sticky and would make k=9 of n=36 a different
population, restarting the rate at n=0 to answer a question software can probably answer.

The RTL exists if software fails: four apertures (221/222/219/223) carrying `paddr[27:20]` and
`[35:28]` for both records, built and lint-clean, exposing `[35:4]` — a 64 GB window, no aliasing
at any size we will run. **The reflash is the project lead's call and is not being requested.**

### XU is a 4 MiB domain, not 2 MiB — the carve recomputed from primary source

The monitor's carve is deterministic and needs no runtime input, so it can be evaluated offline.
Doing so — rather than trusting a source comment that recorded someone's 2 MiB verification —
changes the answer.

From `modcapstone/module/capstone.c:107-111`:

    dom_headroom = max(code_len, DOMAIN_DATA_SIZE) = code_len   (1377704 >> 1536)
    dom_tot_size = code_len * 2 = 2755408 -> 673 pages -> log2 10
    tot_size     = 1024 * 4096 = 4194304 = **4 MiB**

**Independently confirmed by the board:** `DBAS` steps by `0x400000` between consecutive domains
in every transcript — 4 MiB, not 2 MiB.

Feeding that into `sbi_capstone.c:707-720`:

| | 2 MiB (the quoted comment) | **XU, 4 MiB (computed)** |
|---|---|---|
| `repr_gran` | 1024 | **4096** |
| `data_off` | 2048 | **4096** |
| `dom_data.start` | `0x153C00` | **`0x152000`** |

So the site's aliases, as offsets from `DBAS` within a 4 MiB region — and there are **four**, not
three:

| alias | offset | region |
|---|---|---|
| 0 | `0x0aedc0` | code `[0, 0x151000)` |
| 1 | `0x1aedc0` | **dom_data** `[0x152000, 0x400000)` |
| 2 | `0x2aedc0` | **dom_data** |
| 3 | `0x3aedc0` | **dom_data** |

**Three of four aliases are in `dom_data`, one is in code.** Alias 0 is already excluded on the
semantic ground that a capability store into the code region is not a sensible operation, so the
site is in `dom_data` — carved, mapped, owned by the domain, above everything the image placed.
That is consistent with the stack hypothesis and excludes the globals copy the monitor writes to
the *front* of `dom_data`.

**It does not narrow to one address**, which the 2 MiB arithmetic would have appeared to do. The
remaining question is unchanged and needs one number: **where the glue parks `sp`**. If `sp` is
near the region top (`DBAS+0x400000`) the site is deep stack; if just above the image end it is
ordinary stack depth. Those give opposite verdicts from the same address.

### The cap table is excluded, and the stack reading is now the suspicious one

The glue (`silicon-ladder/start-gp-captable-interp.S`, which `build-sqlite-silicon.sh:1818`
confirms is the one XU links) places the cap table at the **top** of `dom_data` and grows the
stack down beneath it:

    :399   ld    s4, 8(s1)      /* count, s1 = blob over dom_data, cursor at base */
    :403   lcc(t1, sp, 4)       /* t1 = sp.END = DBAS + tot_size */
    :404   slli  t3, s4, 4      /* count*16 */
    :406   split(gp, sp, t1)    /* gp = [t1,END) table ; sp = [base,t1) */

`count = 185`, read from `.capstone_gp_initdesc+8` — **not** `.capstone_gp_table+8`, which was a
first misreading. `dom_data`'s front is a copy of the initialised-globals bytes starting at
`GPFREE_GLOBALS_OFFSET` (`sbi_capstone.c:773-775`), and that offset is `0x140000`, matching the
host's printed `Globals offset = 0x140000` and `.capstone_gp_initdesc`'s VA `0x150000`.
Cross-checked: `.capstone_gp_table+0` also reads 185.

So the table is **2,960 bytes** and initial `sp` is at offset `0x3ff470`:

| alias | site | verdict |
|---|---|---|
| k=3 | `0x3aedc0` | stack, **329,392 B (322 KiB)** below initial sp |
| k=2 | `0x2aedc0` | stack, 1,346 KiB below |
| k=1 | `0x1aedc0` | stack, 2,370 KiB below |

**No candidate is in the cap table.** That closes the more interesting of the two branches.

**And it puts us in the regime the RTL lane flagged as suspicious.** Even the shallowest candidate
is 322 KiB down. That is a great deal of stack for this workload, so the honest reading is not
"the site is deep stack" but **"the stack hypothesis is now doubtful"**.

### The assumption that has been carried unexamined

All of the above assumes the faulting address lies **inside the wedging domain's 4 MiB region**.
That has never been established. The mux exposes `paddr[19:4]` and nothing more, so all that is
actually known is *the low 20 bits are `0xaedc0`* — the address could be in the monitor, in
another domain's region, or anywhere else in DRAM. The observed shared regions do not match
(`BASE:82578000`, `8257B000`, `82511000`, `82514000` give low-20 `0x78000`, `0x7B000`, `0x11000`,
`0x14000`), but that is four samples, not a proof.

**This is what makes the upper address bits worth having.** Not to distinguish three stack depths
— to answer whether the site is in the domain at all. The four apertures the RTL lane has built
(`221`/`222` for `paddr[27:20]`, `219`/`223` for `[35:28]`) answer it outright. That is a
well-founded need now rather than a speculative one, and it remains **the project lead's call**,
unrequested.

## Campaign totals, 2026-08-18, `caplifive_s07debug_18august.bit`

| | k | n | rate |
|---|---|---|---|
| `XU` | 16 | 53 | 30% |
| `S7T` | 0 | 23 | 0% |

`S7T`'s zero is **not** evidence about the workload — it returns before `run_sqlite()` is ever
called (see the retraction above). It is a boot control and nothing more.

**Granule samples: 5, every one bit-identical** (`0xaedc0` / `0xaedb0`), across **four distinct
`DBAS`** values — `0x85400000`, `0x80800000`, `0x84C00000`, and `0x85400000` again on a different
boot.

### The halt-control A/B does NOT reproduce the era effect

| arm | k | n | rate |
|---|---|---|---|
| ON | 3 | 5 | 60% |
| OFF | 5 | 13 | 38% |

Fisher one-sided **p = 0.38**, smallest arm 5 against the pre-declared target of 16 — so this is
underpowered and settles nothing on its own. But it is worth recording that the matched design is
**not** showing the era split's 55% vs 15%. The most likely reading is that the era difference was
driven by the confounds named when it was raised — the interleaved ladder arriving at the same
time, and boots 11-13 issuing per-domain halts that timed out — rather than by the halt control.

That is the outcome the pre-declaration was written for: had the analysis been chosen after the
data, the era split alone would have been reported as an instrument-perturbs-subject effect.

### What the site question now needs, and it is one bit of information

Everything computed above assumes the address is inside the wedging domain's region. The mux gives
`paddr[19:4]`; the upper bits decide whether the site is domain memory at all. **No amount of
further board time answers it** — the same low 20 bits appear whether the address is a
domain-relative offset or a fixed absolute address elsewhere, because every observed `DBAS` is
4 MiB-aligned and contributes zeros to that field.

## QUALIFICATION: determinism does not prove the recorded load is the FAULTING load

The granule record **rolls**, and untagged LDC responses are **routine** — both measured today. So
an untagged load unrelated to the fault can overwrite the record between the producing store and
the trap, and **a clobbered record is indistinguishable from a correct one**.

Five bit-identical samples across four `DBAS` values prove the recorded event is *deterministic*.
They do **not** prove it is the *faulting* event. "The last untagged LDC before the trap is always
the same routine one" fits the data exactly as well as "the recorded load is the one that
faulted", and nothing collected so far separates them.

This is a real qualification on the strongest result of the campaign and it was raised by the RTL
lane, not found here. It does not weaken the *adjacency* observation — two records one granule
apart, reproducibly — but it does mean the site has not been shown to be the fault site.

**What would settle it: the producer/consumer correlation bit** — `rd` of the last committed LDC
compared in hardware against `rs1` of the faulting instruction. It is the only reader that says
the recorded address belongs to the fault. It is now the highest-value item in the batch, above
the address bits that motivated the batch in the first place: an unqualified address, however
reproducible, cannot carry a root-cause claim.

# THE FAULTING INSTRUCTION IS NAMED: `sqlite3OsRead+0x4c`

**And it was in the transcripts from boot 1.** The LATCHED TRAP MEPC (switches 196-203) is a full
physical address, and the wedging domain's `DBAS` is printed beside it, so the offset is exact —
**no aliasing, unlike the granule record.** Across every wedge on this bitstream:

| offset from DBAS | count | distinct DBAS | VA (= offset + 0x10000) | symbol |
|---|---|---|---|---|
| `0x2a83c` | **11** | 7 | `0x3a83c` | **`sqlite3OsRead + 0x4c`** |
| `0x138f6c` | 3 | 3 | `0x148f6c` | `output_text + 0xdc` |

    000000000003a7f0 <sqlite3OsRead>:
       3a814: cincoffsetimm a0, s0, -0x30    ; a0 = &stack slot
       3a818: stc  a1, 0x0(a0)               ; STORE the sqlite3_file* capability to the stack
       ...
       3a834: ldc  a0, 0x0(a0)               ; RELOAD it
       3a838: ldc  a4, 0x0(a0)               ; a4 = pMethods
       3a83c: ldc  a4, 0x20(a4)              ; a4 = pMethods->xRead   <-- FAULTS
       3a84c: jalr a4                        ; indirect call through it

The faulting instruction is the load of **`pMethods->xRead`**, and the classic spill/reload shape
sits four instructions above it: a capability `stc`-ed to a stack slot at `+0x28` and `ldc`-ed
back at `+0x44`.

**This is where the project already suspected the defect lived.** The `S7T` selftest was built to
model exactly this — its comment says it plants a value and loads it "the way `sqlite3OsRead`
does", and `decode_s07_cursor` is documented as "the S-07 H1/H2 verdict read from the pMethods
MEMORY SLOT". An independent instrument has now put the trap there.

## And it CONFIRMS the qualification rather than escaping it

The granule record's address does not correspond to this instruction's operand. `pMethods` is a
static vtable, so `pMethods+0x20` should be in `.data`/`.rodata` (`0x1511f0 .. 0x1605a0`), and no
alias of the granule reading `0xaedc0` lands there.

So the two readers **disagree**, which is exactly what the RTL lane predicted: the rolling granule
record was overwritten by a routine untagged load between the producer and the trap. The
qualification is now demonstrated rather than hypothesised, and it is the correlation bit in the
pending batch that would have flagged it at the time.

**Practical consequence: prefer the latched trap mepc to the granule record.** It is a full
address, it needs no alias reasoning, and it has been correct and ignored for twenty-one boots.

## How this was missed for a day

The wedge block printed `[wedge] trap mepc = 0x...` at every wedge and the driver's own comment
said "map onto the domain disassembly to name the faulting instruction". Nobody subtracted `DBAS`.
Attention went to the granule apertures because they were new, while the reader that already
worked sat three lines above them in the same output.

## CORRECTION: the untagged load is `+0x48`, not `+0x4c` — and the readers AGREE

**Retracting the inference that the granule record and the trap mepc point at different events.**
It rested on expecting a record for an access that never happens.

`mcause 25` is **UNEXPECTED_OPERAND**, not INVALID_CAPABILITY — confirmed at the encoder,
`core/anvil_build/capstone_unit.anvilh:303`, in a file that carries this warning at :294-296:

    // showing mcause 25 was recorded as INVALID_CAPABILITY on the strength of this comment, and
    // three [investigations chased the revocation subsystem] for what is an operand
    // type error. 25 is UNEXPECTED_OPERAND. Trust the encoder, not this comment.

So the chain reads:

    3a838: ldc a4, 0x0(a0)     ; a4 = pMethods, read from [a0] = the sqlite3_file OBJECT
                               ;   <-- THIS is the untagged load. It returns a4 with no tag.
    3a83c: ldc a4, 0x20(a4)    ; faults: rs1 (a4) is NOT_CAP. Operand check fires BEFORE
                               ;   any access, so pMethods+0x20 is NEVER READ.

**`mepc` names where the trap was TAKEN (`+0x4c`); the defective load is the instruction that
produced its operand (`+0x48`).** Those are different addresses and the distinction is the whole
correction.

My argument was that `pMethods+0x20` must lie in `.data`/`.rodata` and no alias of `0xaedc0` lands
there, therefore the readers disagree. But no record of `pMethods+0x20` can exist under any
hypothesis, because that access does not occur. The granule record should point at **`[a0]`, the
file object** — and the earlier alias work placed `0xaedc0` outside the image and outside
`sqlite_heap`, in monitor-carved `dom_data`, which is where a stack-resident `sqlite3_file` would
be. The two records one granule apart (`0xaedb0` store, `0xaedc0` load) is an ordinary frame
layout, not a contradiction.

**So the two instruments are plausibly in agreement, and the granule result is not discredited.**

**What survives from the qualification:** the general statement is still correct — the record
*can* be clobbered, untagged loads *are* routine, and the correlation bit is still the right top
item in the batch, because it decides this per-wedge instead of by argument. What is withdrawn is
only the specific claim that *these* readings disagree.

That distinction matters: "the record was clobbered" and "I compared it against the wrong
instruction" have different consequences, and only the first would justify distrusting the granule
data.

**It also strengthens the `S7T` connection.** The selftest is documented as reading "the pMethods
MEMORY SLOT" — that is `3a838`, the load now identified as the defective one, not the vtable
dereference after it. The selftest models the right instruction.

## The write-buffer mechanism: the INLINE shape is excluded, the memset shape is NOT

The RTL lane proposed a candidate mechanism (write buffer per 64-bit **word**, capability tag per
16-byte **granule**, every entry writing the whole granule's tag on drain in round-robin rather
than program order) and asked for one software check: does ordinary compiled code put a **plain
store to `object+8`** before the **`stc` to `object+0`**?

### Result 1 — the inline shape does not occur. Check proven to fire.

Scanned all 333,511 disassembled lines of `XU.dom` for a plain `sd`/`sw`/`sh`/`sb` to `base+8`
followed by an `stc` to `base+0`, requiring **same function**, **≤12 instructions apart**, and
**the base register not redefined in between**:

    STRICT: 0 hits

A first, loose version of the same scan reported **416 hits** — all artifacts of matching on
register *name* across up to 772 bytes of code, i.e. register reuse, not the same object. The
strict result is the real one.

**Positive control, because a zero is not evidence otherwise:** the same scanner run against a
hand-built disassembly containing the exact pattern reports **1 hit**. The check fires.

### Result 2 — the memset shape is NOT excluded, and my check for it was VOID

`memset` is a call, not an inline store, and "zero-init then populate" is the classic shape. A
scan for `stc` shortly after a `memset`/`memcpy` call also returned 0 — **but that zero means
nothing.** Calls in this binary are **indirect**: they render as `jalr a4` / `jalr a1` through a
capability register, and the only symbol-named `memset` references in the whole disassembly are
`memset`'s own internal branches. There are no direct call sites to match, so the scan could not
have fired regardless of the answer.

Caught before reporting. Had it gone out, it would have read as "the memset shape is excluded
too", which is the strongest possible version of the claim and entirely unsupported.

### So the honest verdict

**The mechanism is not confirmed and not killed.** What is established is narrower: *the compiler
does not emit the inline high-half-then-capability pattern anywhere in this domain.* The
`memset`-then-`stc` route remains open and is the one that matters most, since a struct cleared
before its capability field is written is exactly how the condition would arise in ordinary C.

Resolving it needs either the SQLite amalgamation source (not present in the tree at any path
searched) or resolution of the indirect call targets in the disassembly. Neither needs board time.

## The memset shape IS present, in the function that builds the faulting object

Verified at primary source, in this build's own amalgamation copy
(`/tmp/capstone/merge-verify2/obj/sqlite3-capstone.c`; the file is absent from the repo because
it is a **build artifact** — `build-sqlite-silicon.sh:50` takes it from
`$CAPSTONE_TMP_ROOT/sqlite-build/`, which is why an in-tree path search found nothing):

    109864  struct MemJournal {
    109865    const sqlite3_io_methods *pMethod;  /* Parent class. MUST BE FIRST */

    1068    struct sqlite3_file {
    1069      const struct sqlite3_io_methods *pMethods;
    1070    };

    110149  SQLITE_PRIVATE int sqlite3JournalOpen(
    110164    memset(p, 0, sizeof(MemJournal));                                   <-- +16
    110176    pJfd->pMethods = (const sqlite3_io_methods*)&MemJournalMethods;     <-- +28

`pMethods` is the **first** member, so in a capability build it occupies `[p+0, p+16)` — **exactly
one granule**. The `memset` zeroes the whole struct, including `[p+8, p+16)`, the **high word of
that granule**, with plain stores. Twelve source lines later the capability is written to `p+0`
with an `stc`.

That is conditions (1) and (2) of the write-buffer mechanism, in the function that constructs the
`MemJournal`/`sqlite3_file` whose `pMethods` is read by the load at `sqlite3OsRead+0x48` — the
instruction the trap localisation identified as defective.

### What this does and does not establish

* **ESTABLISHED:** the source contains the trigger shape, in the right function, on the right
  object, with the plain store and the capability store close together.
* **NOT ESTABLISHED:** that the `memset`'s entry for `p+8` is still resident in the write buffer
  when the `stc` lands, or that the arbiter drains them in the wrong order on failing runs. Both
  are RTL/timing questions; neither follows from source.
* **NOT ESTABLISHED:** the mechanism itself, which is still under adversarial audit.

### It also explains the rate

The mechanism is deterministic given the same instruction stream **and the same buffer
occupancy** — and occupancy depends on everything that ran before. So the same site can fire on
one rep and not the next **while never moving**: a **fixed site with a variable trigger**, which is
what k=16/n=53 with a bit-identical address actually looks like. Neither a pure race nor a pure
data-dependent bug explains that combination cleanly.

**Status stays neither-confirmed-nor-killed** until the auditor reports and a directed RTL
simulation — `plain store to G+8; stc to G; ldc G` — either reproduces tag loss or does not. That
test is board-free and decides the mechanism independently of whether SQLite triggers it.

# ROOT CAUSE CONFIRMED ON SILICON: the write buffer reorders same-granule stores

Five directed arms, one boot, on `caplifive_s07debug_18august.bit`. 16384 slots per arm.

| arm | sequence | lost | expected | reading |
|---|---|---|---|---|
| wb0 | `stc G` only | **0** | 0 | control clean |
| wb4 | plain `G+16`; `stc G` | **0** | 0 | effect is GRANULE-SCOPED |
| wb3 | plain `G+8`; 64 stores; `stc G` | **0** | 0 | **buffer drained -> no loss** |
| wb2 | `stc G`; plain `G+8` | 15193 (92.7%) | 16384 | detector fires |
| **wb1** | **plain `G+8`; `stc G`** | **1107 (6.76%)** | **0** | **REORDER** |

**Arm 1 loses 1107 tags where program order says none should be lost.** The `stc` writes the
capability to the granule's low word *after* the plain store to its high word, so in program
order the tag is set last and must survive. It does not.

## Arm 1 vs arm 3 is the decisive pair

Identical stores, identical granule, identical addresses. The **only** difference is ~64
unrelated stores between them, which drain the write buffer. **1107 versus 0.** That isolates
buffer co-residency as the variable, which is exactly the hypothesised mechanism and nothing
else. Arm 3's cycle count corroborates that the drain loop really ran: 45,263,573 cycles against
~4,000,000 for the other arms.

## The positive control became a second, independent confirmation

wb2 was expected to lose all 16384 — the plain store legitimately clears the tag. It lost 15193.
**The 1191 survivors are the same reorder running the other way:** the plain store draining
*before* the `stc`, leaving a tag SET on scalar data. So the defect is bidirectional — **tag loss
AND tag forgery** — and the two rates agree:

    arm 1  tag lost      1107 / 16384 = 6.76%
    arm 2  tag forged    1191 / 16384 = 7.27%

Tag forgery is the more serious direction: it fabricates a valid capability over attacker-chosen
scalar data, which is a soundness hole in the capability model rather than an availability bug.

## QEMU is the control that cannot exhibit it

Arms 0/1/3/4 all returned `0xB0000000` — zero loss — under emulation, where a capability store is
one atomic 16-byte-plus-tag operation with no write buffer, no per-word entries and no drain
arbiter. The silicon/emulator difference IS the mechanism, measured against an oracle that
structurally cannot reproduce it. No amount of QEMU testing could ever have found this.

## It explains everything that did not previously fit

* **Deterministic site under a nondeterministic trigger.** Drain order is fixed by the instruction
  stream; buffer occupancy depends on what ran before. Same address every time, fires sometimes.
* **`src` never discriminated.** Both legs deliver the same corrupted value, so it cannot.
* **~7% per opportunity** against a 30%-per-rep SQLite wedge rate — consistent with several
  opportunities per run.
* **The SQLite trigger**: `sqlite3JournalOpen` does `memset(p, 0, sizeof(MemJournal))` and twelve
  lines later `pJfd->pMethods = ...`; `pMethods` is the first member, so the memset's plain stores
  and the capability `stc` land in the same granule. `sqlite3OsRead+0x48` then reloads it untagged
  and `+0x4c` faults with mcause 25.

## Status

**The mechanism is confirmed on hardware.** What remains open is the RTL fix and the provenance
question (whether the `ctag`/`cap_tag_q` path predates the S-06 work), both with the RTL lane, and
one reproducibility boot for this result.

# THE FIX WORKS — and a SEPARATE, TRANSIENT residual is confirmed on silicon

Bitstream `caplifive_s07fix.bit`. All runs `PREFLIGHT=0`, i.e. **ungated** (the overlay is 22 MB
against a 20 MB budget), and labelled as such.

## S-07 is fixed. Every arm, including the one built to catch a bad fix.

| arm | pre-fix | post-fix |
|---|---|---|
| `wb1` plain `G+8`; `stc G` | 1107 lost | **0** |
| `wf1` same, field-checked | 233 lost, 0 corrupt | **0 lost, 0 corrupt** |
| `wf5` scrub + delayed readback | 176 dropped | **0** |
| `wb2` `stc G`; plain `G+8` | 15193 | **16384 — exactly the oracle** |
| `wb0` / `wb3` / `wb4` | 0 | 0 |

**`wb2` = 16384 exactly is a POSITIVE IDENTIFICATION OF THE MECHANISM, not merely a
non-regression.** Option B (forbid co-residency) serialises the two requests, so the plain store
clears the tag every single time — exactly 16384. Option A (granule-aware merge) would have
merged them into one entry and resolved the tag by last-writer-wins inside it, which does not
produce a clean 16384 and would not have needed serialisation at all. The number distinguishes
which fix shipped.

**The corrupted-but-tagged bucket stayed 0**, before and after. It exists only because the naive
fix — propagate the youngest tag — would convert loss into corruption while a tag-only test
certified success. Detector with a demonstrated negative on both sides; whatever shipped is not
that mistake.

## A SEPARATE, PRE-EXISTING, TRANSIENT residual — do NOT blame the fix for it

    wr6  scrub G+8, then type-query IMMEDIATELY   3837 / 3840  (99.92%) LIVE CAPABILITY
    wr7  identical + 300-iteration drain             0 / 3840
    built-in positive control, both arms:   delayed query saw NOT_CAP 3840/3840 -> FIRES

The pair differs by **exactly the delay** — verified in the disassembly, the only
instruction-level difference is the drain loop.

**Mechanism** (RTL lane): the co-residency fix is an ALLOCATION-time check between two write
buffer ENTRIES. This residual needs only ONE entry, so the check never fires and a load never
consults it:

    stc  G, cap     drains to L1, cap_tag_q[G>>4] = 1
    sd   x, G+8     ONE plain entry, word 1, STILL RESIDENT
    ldc  G          granule-aligned -> compares WORD 0, misses the word-1 entry,
                    falls through to the STALE cap_tag_q -> returns a LIVE capability

**Severity, and the distinction that matters:** the pre-fix dropped scrub was **PERSISTENT** — the
capability survived indefinitely. This one is **TRANSIENT**, closing completely once the entry
drains. A program cannot rely on an immediate re-read to confirm it has destroyed a capability;
it can rely on the destruction having happened once the buffer drains.

**Silicon is far worse than simulation here:** Verilator showed 8 of 16 legs, and the trap handler
draining between legs is what made that alternate. On silicon it is 99.92% — essentially always.
The sign was what mattered, and the magnitude is much larger.

### Why `wf5` reported 0 and must not be read as exoneration

`wf5` reads back in a **separate loop** after all 256 slots are stored, so it has a long delay
built in by construction. It measures the same thing `wr7` measures, which is why it reads 0. A
correct measurement of the wrong question — the exact failure class added to this project's rules
the same morning.

## Two caveats on the fix result, both open

* **Provenance unconfirmed.** No `.bit`, no synthesis log, no Vivado log on this machine, and the
  RTL lane's credential 403s on the remote. The attribution rests on: every tree in which `wb1`
  can be 0 carries option B and only option B (`618f4ce36` and earlier have no fix; `5c5f4e3a7`
  through `f231b5af0` are byte-identical in synthesizable logic; options A and C were never
  implemented). That is inference from the result, not a build record.
* **WNS unknown**, retiming left `true`. Four clean 0/16384 results argue against a *gross* timing
  failure, but these arms are narrow and homogeneous — they exercise a thin slice of the design,
  and a marginal path can be data-dependent and untouched by them. An argument, not a timing
  report.

### "PRE-EXISTING" is now MEASURED — it was structural reasoning when first recorded

The residual was written up as pre-existing in `d96fa6c35d7d` **on reasoning, not measurement**.
The pair had only ever run on the FIXED RTL, and a pair run on one revision cannot distinguish
"already there" from "caused by the fix". The RTL lane flagged that themselves and closed it:

    same binaries, pre-fix RTL (a3dbae618) in a worktree

                                       pre-fix a3dbae618      fixed
      s07-wbuf-forward-residual         9 exc /  9234 cyc     9 exc /  9234 cyc
      s07-wbuf-forward-residual-ctl    17 exc / 26361 cyc    17 exc / 26361 cyc

Identical in both arms: the fix neither repairs the residual nor worsens it.

**The A/B carries its own model-identity control**, which is what makes identical numbers mean
something. Identical results are also exactly what running the same model twice produces, so a
worktree silently reusing the fixed build would have printed this table and looked like
confirmation. `s07-wbuf-tag-reorder` in that worktree gives 4 exceptions at 9150 cycles — the
PRE-FIX signature — against 1 at 9138 on the fixed model, plus zero occurrences of
`gran_conflict` in the worktree source before the run. Committed `dc283fbab`.

### "Transient" must not be read as "edge case"

The window is short-lived but, **while the entry is resident, very nearly certain: 3837 of
3840, 99.92%.** Short-lived is not the same as unlikely. The operational consequence is
unchanged — a program cannot trust an immediate re-read to confirm it has destroyed a
capability, but can trust the destruction once the buffer drains — and the word "transient"
should not be allowed to carry more reassurance than 99.92% supports.

Quote the silicon number, not the Verilator one. The 8-of-16 seen in simulation is a harness
artifact: the trap handler runs between legs and drains the write buffer, resetting the phase,
which is what made that pattern alternate. It was never a probability.

### Aperture note — CORRECTED. The provenance aperture is switch 237, NOT 205.

**Switch 205 discriminates nothing** and would have produced a plausible-looking reading that
cannot answer the question — the worst kind. Resolved from source across all four candidate
trees, bank 111 reg `5'b01101` = **switch 237**:

| tree | switch 237 |
|---|---|
| `618f4ce36` | `commit_instr_id_commit[0].pc[63:56]` |
| `5c5f4e3a7` | **`8'hA5`** |
| `6175ea654` | **`8'hA5`** |
| `f231b5af0` | `commit_instr_id_commit[0].pc[63:56]` |

    read SWITCH 237
      0xA5                        -> instrument-carrying tree, 5c5f4e3a7..6175ea654
      varying commit-PC high bits -> 618f4ce36 or the stripped f231b5af0

An earlier version of this note said 205, from a derivation that took the offset 192 of four
sample arms (219/221/222/223) — **all of them in bank 110** — and applied it to an arm in bank
111. A sample uniform in the variable not being controlled for.

A claim that "237 is impossible because the case value is 5 bits, so the highest reachable
switch is 223" is **wrong**, and only true within one bank. 237 is not merely reachable — it is
the aperture that carries the answer. `debug_byte_sel` selects among eight
banks of 32: bank 110 spans 192-223, **bank 111 spans 224-255**. 237 is bank 111 reg 13 and is
perfectly reachable — empirically, `sw=230` reads `commit pc[7:0]` in the boot-4 transcript.
Whether a generation marker lives there is a separate question about which tree was synthesized;
reachability is not the objection.

## PROVENANCE RESOLVED: the bitstream is `f231b5af0`, the STRIPPED tree

Read halted on `caplifive_s07fix.bit`, resident bitstream confirmed by `nvbit()`:

    sw 237 = 0x00        the discriminator
    sw 236 = 0x00
    sw 230 = 0x7c        internal control -- three apertures, two distinct values, so the
                         mux IS being addressed and 0x00 is a reading, not a stuck path

**Deduction, combining two independent measurements:**

1. `237 = 0x00`, and `8'hA5` is a **constant** in the instrument trees, so the bitstream is
   **not** `5c5f4e3a7` and **not** `6175ea654`. It is `618f4ce36` or `f231b5af0`, both of which
   carry `commit_instr_id_commit[0].pc[63:56]` there — and `0x00` is exactly the high byte of a
   low or machine-mode PC.
2. `wb1` = 0, from 1107, **requires option B**, which rules out `618f4ce36` — that tree contains
   no fix at all.

**Therefore the bitstream is `f231b5af0`: option B present, instrument absent.**

### This CORRECTS the timeline assumption, and the attribution survives it

The prediction was `0xA5`, on the reasoning that the strip landed at 16:58 and `f231b5af0` at
17:53, both *after* the flash. The reading says otherwise: the flash came from the stripped tree,
so it happened after `f231b5af0`, not before.

**Nothing about the fix result changes.** The attribution never depended on the aperture — it
rests on `wb1` being 0 in a tree where only option B exists — and `f231b5af0` carries option B
with synthesizable logic byte-identical to `5c5f4e3a7`. The S-07 fix validation stands exactly as
recorded.

**What it does change:** the earlier note that these results came from an instrument-carrying
tree is wrong. The bitstream has **no** S-07 instrument, which also explains why every aperture
in the 230-237 range reads as commit-PC bytes rather than as the batch's readers. Any future
attempt to read the granule addresses or the correlation bit **on this bitstream** will fail, and
should not be diagnosed as a broken instrument.
