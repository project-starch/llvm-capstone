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

**Do not quote the cold slope as a constant.** It rests on one segment; 3.000 rests on six.

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

## Caveats that belong with the numbers

- The spliced cost is flat **in N**, not immune to cache pressure: at and past capacity its own small
  fixed working set goes cold and the figure moves between 328 and 777 with **no trend in N** (777 at
  2,048, 516 at 3,072 — non-monotone, i.e. noise).
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
- **The live trap:** any value ≡ 0 mod 16 loads a zero counter and realises as ~2 cycles — asking for a
  32- or 48-cycle memory gets you none. Only `FixedDelay == 0` reaches the true bypass, and `1` is
  special-cased. **Usable range is 2..15.** No value other than 40 has ever been used, so nothing in the
  record is affected by that trap today.

**This is a parameter-label correction, not a retraction.** Every finding that rests on "non-zero memory
latency changes the behaviour" stands — S-12's store-buffer result, R-26's deciding arm, R-34's
delay-invariance check. Only the magnitude was mislabelled: those runs had an 8-cycle memory, not a
40-cycle one.
