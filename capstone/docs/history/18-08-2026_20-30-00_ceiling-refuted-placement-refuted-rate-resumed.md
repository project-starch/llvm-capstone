# Three results: no run ceiling, placement refuted, and S-07 is reproducing again

Date: 2026-08-18. Bitstream `caplifive_s06s08fix_s07tag2_618f4ce.bit`, compiler with C-19 reverted
(`XU` = `f1214600d0dac351`, byte-identical to the historical reproducer).

## 1. The "~6-run ceiling" does not exist

`SILICON-BLOCKER.md` records a ceiling at ~6 domain runs per boot, attributed to an unknown
exhausted resource, and it has shaped every experiment since: batches were sized to 4-5 domains
because more was believed impossible.

**Measured: 10 identical `k800` rungs in one boot, 10 of 10 PASS.** No failure, no degradation.
The SQLite path in the same session ran a control plus three full `XU` reps before an unrelated
S-07 wedge ended it — i.e. it did not hit a ceiling either.

Consequence: reps per boot were never the constraint, so **rate measurement is cheap** and the
plan's Priority 1 (raise `CAPSTONE_MAX_REGION_N`, warm-boot the board to skip the JTAG upload) is
**unnecessary**. Nothing needed changing; we had simply never asked for more than five.

## 2. Physical placement is REFUTED as the modulator

`DBAS` is the domain's true physical address and varies per boot. The hypothesis was that S-07
depends on where the domain lands. A retrospective harvest over every recorded run with a DBAS
kills it, at zero board cost:

| run | time | domain | hash | DBAS | position | outcome |
|---|---|---|---|---|---|---|
| `close1` | 03:42 | `XU` | `f1214600` | `0x84400000` | 4 of 4 | **WEDGED** |
| `close2` | 03:50 | `XU` | `f1214600` | `0x84400000` | 4 of 4 | **PASS** |

Same domain, same hash, same physical address, same position, eight minutes apart, no reflash.
Every recurring DBAS slot appears on both sides of the ledger. Position-in-boot shows no pattern
either, and is structurally censored anyway because a wedge ends the session.

The same harvest found that the `S7T` "wedge" chased earlier today was contaminated: no
`SQ: G/enter`, mcause 29, and an **identical mepc across two independent boots** — a stale latch
from before the domain ran, not a result.

## 3. S-07 is reproducing again, and it flips WITHIN a boot

Today's data for `XU`, one hash, one bitstream:

| boot | reps | outcome |
|---|---|---|
| 16:19 | 4 | PASS, PASS, PASS, PASS |
| ~20:00 | 3 | PASS, PASS, **WEDGE** (entered, so genuinely S-07) |

**k = 1 wedge in n = 7.** Two facts follow. First, the defect had not gone away this morning; the
4/4 was a sample, not a cure — which is exactly why the 4/4 was recorded as "unmeasurable" rather
than "fixed". Second, and new: the flip happened **inside a single boot**, reps 2-3 passing and
rep 4 wedging. That weakens any explanation that is purely per-power-up (DRAM calibration,
thermal state at boot) and points at something that drifts or accumulates during a session.

## What to do with this

* Keep accumulating k/n opportunistically — it is now nearly free, and no historical "X wedges"
  claim should be trusted until it is restated as a rate.
* The next discriminator worth spending a boot on is `tagsweep` (the non-wedging tag-loss counter)
  run **alongside** `XU` in the same boot. If a boot where `XU` wedges also shows non-zero tag
  losses, `tagsweep` becomes a fast, non-destructive rate meter and the defect is a bulk memory
  effect after all; if `XU` wedges while `tagsweep` reads zero, that is refuted cheaply.
* Do NOT spend effort on the ceiling or on placement. Both are closed by the above.
