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
