# S1 — revoke cost against the number of dead nodes the walk crosses, spliced vs unspliced

**Date:** 2026-09-16.
**Trees:** `wt-probe` = `r30-r31-init-revoke` `4cc068572` (unspliced control);
`wt-splice` = `r12-splice-revoked-nodes` `f1331daed`.
**The pair is clean by construction:** `4cc068572` **is** the merge-base of the two branches, the only
two commits between them are the splice and its measurement, and the only differing source file under
`core/` is `capstone_rev_node.anvil` (40 changed lines). Both trees passed
`capstone/tests/anvil-staleness-check.sh` before the run, and the presence of the splice was confirmed
**by content** in the generated RTL (`serving_*` appears in `wt-splice/core/capstone_rev_node.anvil.sv`
and nowhere in `wt-probe`), not by timestamp.

**Instrument:** Verilator, `S12_MEM_DELAY=12`, read back from
`work-ver/Variane_testharness__verFiles.dat` on every rung — never from the log.
**Fixture:** `verif/tests/custom/capstone/r12-s1-walk-ladder.S`, `NROUNDS` set per build. MREV on one
linear base inserts each handle immediately above the base, so the first is outermost. Four live nodes,
then a middle handle, then `NROUNDS` more below it. Revoking the middle kills those `NROUNDS`; the
**timed** revoke is then issued on the outermost, whose walk must cross the live nodes and then the
corpses.

## The reading, written down before the run and obtained

> unspliced cost rises with NROUNDS; spliced cost is flat in NROUNDS.

| N dead nodes crossed | unspliced | spliced | ratio | regime |
|---:|---:|---:|---:|---|
| 1 | 65 | 71 | 0.9× | |
| 8 | 273 | 328 | 0.8× | |
| 64 | 441 | 328 | 1.3× | |
| 128 | 633 | 328 | 1.9× | |
| 160 | 729 | 328 | 2.2× | |
| 512 | 1,785 | 328 | 5.4× | chain = 25 % of dcache |
| 1,024 | 3,321 | 328 | 10.1× | 50 % |
| 2,048 | 7,632 | 777 | 9.8× | **100 % — at capacity** |
| 3,072 | 96,822 | 516 | **187.6×** | 150 % — cold |

**Unspliced marginal cost is exactly 3.000 cycles per dead node.** The fit `rev2 = 3N + 249` is exact,
not approximate, at every rung from N=8 to N=1,024 — six consecutive segments. **Spliced marginal cost
is exactly 0.000:** 328 cycles at N = 8, 64, 128, 160, 512 and 1,024, identical to the cycle.

The splice costs a fixed ~79 cycles and removes a per-node 3. **Crossover at N ≈ 26**: below that it is
a small loss, above it the saving grows without bound.

## The cache boundary falls exactly where the geometry predicts

The dcache is 32,768 B with 128-bit lines and a `rev_node` slot is 128 bits — **one node per line,
2,048 nodes to fill the cache.** The 3.000 slope holds through N=1,024 (50 % of capacity), bends at
N=2,048 (100 %, +1,239 cycles over the warm fit) and breaks completely at N=3,072.

**This makes 3 cycles/node the L1-HIT cost, and therefore a LOWER BOUND on what the splice saves.**
One SQLite speedtest1 run mints 43,355 nodes = 677 KB = **21× the dcache**, so the workload regime is
entirely on the far side of that boundary. The simulation understates the benefit; it cannot overstate it.

### The 12.7× jump at N=3,072 survives its own control

A single surprising rung is not a result. The internal control is `rev1` — the revoke that *kills* the
nodes — which walks the same chain on the same tree in the same run:

| segment | rev1 cycles/node | rev2 cycles/node |
|---|---:|---:|
| 512 → 1,024 | 60.0 (warm baseline) | 3.0 (warm baseline) |
| 1,024 → 2,048 | 60.4 | 4.2 |
| 2,048 → 3,072 | 115.6 (**+55.6**) | 87.1 (**+84.1**) |

Both revokes gain a per-node penalty **of the same order** in the cold segment, which is what a cache
effect looks like. rev2's 29× is a small-base artifact — its warm cost was only 3. Had rev1 stayed flat
while rev2 exploded, the reading would have been about rev2's path, not about the cache.

**⚠ RETRACTED 2026-09-16, same day — 87.1 IS NOT THE COLD SLOPE.** It is worse than thin, and the
defect is the one this very note warns about two sections down. N=2,048 sits only 1,239 cycles above
the warm fit — **84 % of that point is still warm** — so the 2,048 → 3,072 segment *crosses the
boundary*, and the slope of a segment spanning a knee measures the knee. A fit range that straddles a
regime change was exactly the error the board lane independently diagnosed in P1 the same afternoon;
mine had it too, and I wrote the caution about fit ranges without applying it to my own table.

Use the board lane's Q4 figure instead: **25.45 cycles per node walked, R² = 0.999969 over eleven
points, the whole fit inside the cold regime** at 25× dcache capacity. As a sanity check in the same
direction, *total*/N at my one cold point is 31.5 cyc/node — the same order, where the marginal 87.1
is not. The rows below are kept only to show the crossover; **no slope may be taken from them.**

## What the timing alone could NOT distinguish

A flat cost is *also* exactly what a revoke 2 that terminated early and did nothing would produce. The
timing cannot separate those two, so it was not asked to. A `WITNESS` build — behind `#ifdef`, leaving
the timed path byte-identical to the ladder already run — samples the middle handle's own node validity
either side of revoke 2. Revoke kills the subtree *below* a handle, so s1's node survives revoke 1 and
must be killed by revoke 2:

| tree | s1 valid before revoke 2 | after |
|---|---:|---:|
| spliced | 1 | 0 |
| unspliced (control) | 1 | 0 |

Revoke 2 does its job on both trees. **The flat cost is a working splice, not an early exit.**

**The `#ifdef` claim is not taken on trust.** The witness blocks were added *after* the warm ladders
ran, so the committed fixture is not literally the file that produced 65…729 and 71…328. The proof is
in the artifact, not the source: the deep rungs were compiled from the **post-edit** source without
`-DWITNESS` and reproduce the pre-edit fit exactly — 1,785 and 3,321 on `3N + 249`, and 328 on the
spliced side, to the cycle. A timed path that had moved could not do that.

## Caveats that belong with the numbers

- The spliced cost is flat **in N**, not immune to cache pressure: at and past capacity its own small
  fixed working set goes cold and the figure moves between 328 and 777 with **no trend in N** (777 at
  2,048, 516 at 3,072 — non-monotone, i.e. noise).
- **The region below N=8 is a separate regime with a measured mechanism** — see the section below; it
  touches neither the slope, the spliced 0.000, nor the crossover. (Superseded caveat: N=1 is ~200
  cycles BELOW the fit on BOTH trees) (65 against 252; 71 against 328) and is excluded
  from every slope quoted here. It is common-mode — the two trees step by 4.2× and 4.6× between N=1
  and N=8 — so it cannot touch the spliced-vs-unspliced comparison. But a 4× step in revoke 2's
  *fixed* cost, over a range where the spliced walk does identical work at both ends, is
  **unexplained**, and "cold start" is a label rather than an explanation. Possibly a lead about
  REVOKE's entry cost; not chased here.
- Simulation, not silicon. This measures the mechanism, not a workload.
- Every rung on both trees: 0 traps, 0 `Exception:` lines, construction cause 0, both revoke causes 0,
  the killed run reading invalid and the outermost reading valid. The fixture reports; it does not judge.

## A correction to a number stamped across the docs: `S12_MEM_DELAY` is not a cycle count

`stream_delay.sv` (both copies in the tree) declares `CounterBits = 4` and
`assign counter_load = FixedDelay`, so the delay parameter is **truncated to its low four bits**.

- `S12_MEM_DELAY=40`, which appears in **39 places** across the docs and testlists and is described as
  "a 40-cycle memory", realises as **40 mod 16 = 8**.
- Confirmed behaviourally, not only by reading source: on one tree, define 12 and define 28 (28 mod 16
  = 12) produced **identical** rev1, rev2 and total cycle counts, while the artifact readback proved the
  two builds really did receive different defines.
- **The live trap, MEASURED not inferred:** any value ≡ 0 mod 16 loads a zero counter and realises as **less delay than define 2** —
measured at define 16 on the same tree and test: rev2 = 51, **identical to the true-bypass run**, total
1,004 against 708 at true bypass and 1,415 at define 2 (define 12 gives 3,427). Asking for a 32- or
48-cycle memory gets you essentially none, and it reads as a clean negative. Only `FixedDelay == 0` reaches the
  true bypass, and `1` is special-cased. **Usable range is 2..15.** No value other than 40 has ever
  been used, so nothing in the record is affected by that trap today.

The 4-bit counter is not new: the tracked copy dates from 2022 (`8a5898dce`) and reads
`CounterBits = 4` at `7e4dc440f` (2026-09-03, the S-12 run) and at `ef5a8eaf2` (2026-09-08, the R-26
run), so the relabel applies to those runs as fact rather than inference.

**AND THE KNOB IS NOT MONOTONE — this is the part that bites.** Truncation on its own reads as
"the number is smaller than you thought", which leaves the reader's mental model intact and still
wrong. The delay is a **period-16 sawtooth in the define, not a dial.** Measured on one tree and test
(total cycles): define 0 → 708, define 2 → 1,415, define 12 → 3,427, define **16 → 1,004**. Turning
the knob *up* from 12 to 16 turns the latency *down* to near bypass. Anyone reasoning "larger define,
more latency" gets a plausible-looking result rather than an obvious failure.

**This is a parameter-label correction, not a retraction.** Every finding that rests on "non-zero memory
latency changes the behaviour" stands — S-12's store-buffer result, R-26's deciding arm, R-34's
delay-invariance check. Only the magnitude was mislabelled: those runs had an 8-cycle memory, not a
40-cycle one.

## The sub-8 anomaly, chased to its mechanism

The first write-up excluded N=1 as "cold start" and called the region below N=8 an unexplained step.
Both descriptions were wrong, and the real shape is sharper.

| N | unspliced rev2 | `3N + 249` | excess |
|---:|---:|---:|---:|
| 2 | 59 | 255 | −196 |
| 3 | 107 | 258 | −151 |
| 4 | 201 | 261 | −60 |
| 5 | 264 | 264 | 0 |
| **6** | **358** | 267 | **+91** |
| **7** | **361** | 270 | **+91** |
| 8 | 273 | 273 | 0 |
| 9, 10, 12, 16 | 276, 279, 285, 297 | — | 0, 0, 0, 0 |

**Exactly two rungs sit exactly +91 cycles above the fit, and they are N = 6 and 7.** Everything from
N=8 up is on it — ten exact points now (8, 9, 10, 12, 16, 64, 128, 160, 512, 1,024). Reproduced across
separate runs and separate testlist files, value for value.

**The mechanism is the write-through dcache write buffer, and this is measured, not inferred.**
`CVA6ConfigWtDcacheWbufDepth = 8`. Halving it to 4 and re-running moved the +91 pair from {6, 7} to
**{2, 3}** — a shift of exactly 4 — while **every rung from N=8 upward stayed byte-identical** between
the two configurations. So the anomalous pair sits at `depth − 2` and `depth − 1`, tracks the buffer
depth exactly, and the change perturbs nothing in the region the headline fit rests on. The control
tree was then restored and re-verified by reproducing N=8 = 273.

Below `depth − 2` there is a third regime, rising at ~60 cycles per node — the same 60 that `rev1`
pays per node it kills, i.e. the cold-access cost. At the default depth its line crosses the slope-3
line at N=5, which is why N=5 lands on the fit by coincidence rather than by being in the fitted
regime.

**And extrapolating the fit downward fails in BOTH directions, so it is not a conservative move
either way.** Against the measured values: it over-estimates by 196, 151 and 60 at N = 2, 3, 4, and
under-estimates by 91 at N = 6 and 7. It is not a safe lower bound on small-revoke cost and not a safe
upper bound — it is simply out of range. Anyone wanting the cost of a revoke that crosses fewer than
eight dead nodes has to measure it, and should expect the answer to depend on the write-buffer depth.

**What this changes: nothing in the result, and one thing in how the intercept may be quoted.** The
3.000 slope, the spliced 0.000, and the crossover at N ≈ 26 are all inside the fitted range and
untouched. But `249` is the intercept **of a fit valid for N ≥ 8**, not a fixed cost of revocation, and
it should never be quoted bare.
