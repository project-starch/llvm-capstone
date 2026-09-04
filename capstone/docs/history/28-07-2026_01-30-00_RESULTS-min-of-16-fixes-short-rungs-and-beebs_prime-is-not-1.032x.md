# min-of-16 gives one certifiably clean row — and shows `beebs_prime` is NOT 1.032×

**Date:** 2026-07-28
**Lane:** C (primary)
**Cost:** 1 board boot (16 passes × 6 rungs). Board off + unlocked.
**Headline:** the paper's flagship "3.2 % scalar" number is an artifact of a contaminated
baseline. The real figure is **≥1.6×**, and it is not yet certified.

---

## What was run

Each baseline rung measured **16 times** in one boot. Interrupts arrive on a timer, not in
step with the kernel, so the pass with the **minimum instret** is the least-disturbed run and
the number of passes **tied** at that minimum is the evidence it is genuinely clean.
`ctrsanity` is the calibration: its true interrupt-free value is known independently from the
domain side (6.003 cyc/iter).

| rung | tie at min instret | cycle spread | old ratio | **new ratio** | Δ |
|---|---|---:|---:|---:|---:|
| `beebs_bs` | **15/15** | **45** | 1.181× | **1.274×** | +0.093 |
| `beebs_prime` | 5/15 | 70,757 | 1.073× | **1.605×** | **+0.531** |
| `beebs_cnt` | 1/15 | 70,378 | 0.773× | **1.165×** | +0.392 |
| `rv8_primes` | 1/15 | 85,832 | 1.051× | 1.055× | +0.003 |
| `ctrsanity` | 1/15 | 194,692 | 0.824× | 0.824× | −0.000 |

## Three findings

### 1. `beebs_prime` at 1.032× is wrong, and the error is large

Its baseline was heavily contaminated. Min-of-16 drops it from 44,510 to **29,775 cycles**
(−33 %) and from 14,479 to **12,558 instructions** (−13 %) — the old "warm" pass was carrying
~1,900 instructions of interrupt handler.

The published ratio **1.032× becomes ≥1.605×**. This is the paper's flagship scalar number and
the basis of the "pervasive spatial safety costs 3.2 %" headline. **That claim cannot be made.**

It is not yet a replacement figure either: only **5 of 15** passes tied at the minimum, so even
this baseline is probably still disturbed, and the true ratio is likely **higher** than 1.605×.

### 2. `beebs_cnt`'s impossible ratio is explained — it was interrupts

0.773× → **1.165×**. The sub-1.0 ratio that made `cnt` unpublishable was baseline
contamination, not a mystery. It is now a sensible number, though at 1/15 ties it is not
certified.

This also closes the loop on why the old cleanliness test failed: `cnt` twice showed
byte-identical instret across two passes and was still contaminated. **Two passes can take the
same number of interrupts and both be dirty.** Sixteen passes catch what two cannot.

### 3. `beebs_bs` is the first certifiably clean row

**15 of 15** passes tied at minimum instret with a **45-cycle** spread across the whole set.
That is what an interrupt-free measurement looks like, and nothing else in the ladder has ever
shown it.

| | capability | baseline (clean) | ratio |
|---|---:|---:|---:|
| cycles | 2,258 | **1,772** | **1.274×** |
| instructions | 875 | 827 | **1.058×** |

## The method's limits, measured rather than assumed

`ctrsanity` exists to answer exactly this, and the answer is blunt: **min-of-16 recovered
essentially nothing** for it — 7.290 cyc/iter against the old 7.287, where the true clean value
is **6.003**. At ~15 ms per pass and a ~1 kHz tick, no pass can avoid interrupts, so taking a
minimum over more of them does not help.

So the method works as a function of kernel duration:

| kernel length | verdict | example |
|---|---|---|
| ≲ 2 k cycles | **fully clean**, certifiable | `beebs_bs` (15/15) |
| ~10 k–170 k | large correction, **not certified** | `beebs_prime` (5/15), `beebs_cnt` (1/15) |
| ≳ 700 k | **no help at all** | `ctrsanity`, `rv8_primes` (1/15) |

**`rv8_primes` is therefore still uncorrected and still wrong.** Its 1.055× carries the full
~1.21× environment penalty, so its true overhead is plausibly ~1.28× — but that is an inference
from the calibration, not a measurement, and must not be published as one.

**The bare-metal baseline remains necessary** for every kernel above ~700 k cycles. What has
changed is that it is no longer necessary for *all* of them.

## Bonus: a mismatched −O level found in the baseline spec

`build-ladder-base-fpga.sh` listed `beebs_recursion` at **−O0** while its own comment three
lines below said the baseline "has to be −O1 as well", and the published pair is −O1. The −O0
baseline returns 19,825 cyc / 4,759 instr against the published 10,523 / 2,019 — about 2×.
Anyone running the baseline without `LADDER_OPT` was silently producing a mismatched
denominator. Fixed to −O1.

This is the same family as I-1 and was found the same way: by a number that did not match a
documented one.

## Consequence for the paper

**Do not publish the current spatial-overhead table.** Specifically:

- `beebs_prime` 1.032× is **withdrawn** — it is ≥1.605× and not yet certified.
- The "3.2 % scalar / 5.0 % array / 80 % recursive" headline is **withdrawn**; its cheapest and
  most quotable component was the most contaminated.
- `beebs_bs` at **1.274× cycles / 1.058× instructions** is the one row that can be defended
  today, and it is clean by an explicit, reportable criterion.
- Instruction ratios remain far more trustworthy than cycle ratios throughout.

The direction of every correction so far is the same: **our overheads are larger than we
claimed.** Three separate baseline-environment confounds have now been found (cold/warm paging,
timer interrupts, and a mismatched −O level), and all three flattered us.
