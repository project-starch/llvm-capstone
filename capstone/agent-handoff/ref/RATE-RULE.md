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

Reps are cheap: **10 identical rungs ran in one boot with 10 passes**, so the "~6-run ceiling"
recorded elsewhere does not exist and never constrained anything. There is no excuse for an n of 1.

Accumulate k/n opportunistically by appending reps to boots that are happening anyway, and
classify failures before counting them — an S-07 wedge dies *after* `SQ: G/enter`; a resource or
entry-stall failure dies before it, and must never be counted as a defect.
