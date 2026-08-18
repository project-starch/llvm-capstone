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
