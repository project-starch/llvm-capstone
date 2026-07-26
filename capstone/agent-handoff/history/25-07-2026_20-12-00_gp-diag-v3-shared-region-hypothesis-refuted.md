# BOARD RESULT v3: the shared-region hypothesis is REFUTED — and so is "loops"

**Date:** 2026-07-25 · **Lane:** B · **Third board run** (gp_diag3, one power-cycle).
Refutes the leading hypothesis of
`25-07-2026_18-44-49_gp-diag-v2-all-probes-correct-fault-is-shared-region-loop.md`.
Board left powered off + unlocked.

## Result

```
rung       retval       oracle       cycles   correct
gp_diag3   1341388517   1967218313   12119    NO

dbg0=65280      dbg1=2308734352 dbg2=2174516624 dbg3=2048
dbg4=134282752  dbg5=65280      dbg6=2308734352 dbg7=134218537
dbg8=12648430
```

| slot | probe | expect | silicon | |
|---|---|---:|---:|---|
| dbg0 | A loop over a **global** array (gp cap-table) | 65280 | 65280 | ✅ |
| dbg1 | B loop over **res[]** (shared region) | 65280 | 2308734352 | ❌ |
| dbg2 | C **straight-line** read of res[] — *no loop* | 65280 | 2174516624 | ❌ |
| dbg3 | D loop over res[] at a **constant** index | 2048 | 2048 | ✅ |
| dbg4 | E loop over a **local stack array** — *no res* | 65280 | 134282752 | ❌ |
| dbg5 | F loop **storing** into res[], straight-line readback | 65280 | 65280 | ✅ |
| dbg6 | G loop over res[] via a walking pointer | 65280 | 2308734352 | ❌ |
| dbg7 | H nested loop over res[] (the v2 fold's shape) | 255 | 134218537 | ❌ |
| dbg8 | I canary | 0xC0FFEE | 0xC0FFEE | ✅ |

## Two more hypotheses dead

1. **"Iterated access through the shared-region capability" — REFUTED.** `dbg4`
   loops over a **local stack array**: no shared region, no global, no cap-table.
   It is wrong. The fault does not need the region.
2. **"It needs a loop" — REFUTED.** `dbg2` is a straight-line sum of eight
   `res[W1+k]` reads with constant indices and no loop at all. It is wrong.

Note these also retro-refute the v2 write-up's framing: v2 concluded the fault was
"confined to iterated access through the shared-region capability" because that was
the only construct left standing. It was the only one v2 *tested*.

## What the numbers say (INFERENCE from sums — not established)

Each probe reports a **sum over 8 reads**, so a wrong probe conflates eight
unknowns; this is the same weakness that made v1's checksum useless. Still, two
decompositions are clean enough to be worth recording:

- **E**: `134282752 = 0x08000000 + (seeds 1..7)`. Read as: `larr[0]` returned
  **0x08000000** (2^27) instead of 0x100, the other seven elements correct.
- **B/C**: assuming the same shape, `res[W1+0]` returned **0x899B7F90** (loop) and
  **0x819B7F90** (straight-line). Both look like DRAM addresses (base 0x80000000),
  and they differ by **exactly 0x08000000** — the value E saw.

A 128 MiB length and two DRAM addresses is the signature of **capability metadata
reaching a data load**, not arithmetic corruption. That would also explain why
every prior "wrong checksum" resisted single-word inversion: the injected values
are addresses, nowhere near the plausible-data space those solvers searched.

**Treat this as a lead, not a result.** It rests on assuming the fault hits element 0
of each window, which the sums cannot confirm. `dbg3` (eight reads of `res[W1+0]`,
all correct) sits awkwardly with it and is not yet explained.

## Also new: a real QEMU parity leg

v1/v2 were "QEMU-validated" through the stock `capstone-test` `call_dom` path,
where the domain's `res` argument is a pointer to a **single `unsigned` on the
monitor's stack** — there is no shared region at all. That path could not have
exercised the construct v2 accused, and writing `res[3..]` through it would have
scribbled on the monitor. So "QEMU-correct, silicon-wrong" was, until now, a
comparison across two different harnesses.

`rtl-smoke/run-ladder-perf-qemu.sh` (new) runs the **board's own controller**
(`ladder_perf_ctl`, real 4 KiB shared region, share-is-the-entry) under QEMU.
gp_diag3 passes there with all 9 probes correct and `retval == oracle`, so the
divergence is now a controlled statement about silicon.

## Method notes worth keeping

- **Seeds are distinct powers of two** (`256<<i`) so every expected value has a
  unique binary decomposition and a wrong sum names which elements were added.
  That is what made E readable at all. Keep this; never fold diagnostic data.
- It was still not enough: the bad reads are not seeds, so subset-sum decoding
  leaves an "unexplained" residue. **Report raw words, not sums** — hence v4.
- v3 introduced an uncontrolled difference from the clean v2 run: its work lived
  in a `noinline` helper taking the region **capability as an argument**, which v2
  never did. v4 uses `always_inline` to put the body back in `domain_main`.

## gp_diag4 (two board runs): CLEAN — which is inconclusive, not exculpatory

v4 has no loops at all: seed three windows straight-line (shared region, local
stack array, global array via the gp cap-table), dump **every element
individually**, then re-read the region window a second time. 33 slots.

**All 32 words correct on silicon, `retval == oracle`, both times.** v4 does not
reproduce the fault. It has no positive control, so this rules nothing out — it
only says the minimal straight-line form is clean. Recorded as a negative result,
not as evidence that the earlier failures were spurious.

## Replication: the fault is DETERMINISTIC

`gp_diag3.dom` was re-run **byte-identical** (sha `158760ea74175579`, verified
against the first run's transfer log — a true replication, not a rebuild):

| | run 1 | run 2 |
|---|---|---|
| retval | 1341388517 | **1341388517** |
| dbg0..dbg8 | 65280, 2308734352, 2174516624, 2048, 134282752, 65280, 2308734352, 134218537, 12648430 | **bit-identical** |
| cycles | 12119 | 12166 |

Every wrong word reproduces exactly; only the cycle count moves (47). This was a
control I should have run before v3, not after: three hypotheses had already been
built and demolished on the assumption that the phenomenon was stable, and that
assumption was never tested. It holds — but it was luck, not method.

## The decisive datum: the STORES LANDED  — **RETRACTED 2026-07-26**

> **This section's conclusion is WRONG and withdrawn.** It generalised from
> `W1[0..3]` (all this run could see) to "the stores land". A 45-slot re-run of
> the *identical* dom shows **`W1[7]` holds a DRAM address, not its seed** — there
> IS a store-side component, and the whole-window "load-side fault" claim does not
> survive. The caveat below correctly identified `W1[4..7]` as unobserved; the
> conclusion should never have been stated ahead of observing them.
> See `26-07-2026_02-21-26_gp-diag-two-fault-model-load-side-claim-retracted.md`.
> The rest of this note (the refuted shared-region/loop hypotheses, determinism,
> the QEMU parity leg) stands.

Raising `LADDER_DBG_SLOTS` to 33 made the controller print `res[32..36)`, which is
gp_diag3's seeded data window `W1[0..3]` — read by the **host, after the domain
returned**:

```
dbg29=256  dbg30=512  dbg31=1024  dbg32=2048      <- exactly the seeds
```

So memory holds the **correct** values, while the domain's own straight-line read
of that same window (probe C) summed to garbage. For `W1[0..3]` this is therefore
a **load-side fault inside the domain**, not a store that never landed and not
corruption of the region's contents.

Caveat, stated precisely: this covers `W1[0..3]` only. `W1[4..7]` are `res[36..40)`,
past the 33 printed slots, so they are still unobserved — and a wrong value there
would explain probes B/C/G/H (which read all eight) failing while D (which reads
only `W1[0]`, eight times) passes.

## Where this leaves the root cause

Established: deterministic; needs neither a loop nor the shared region; not the
gp cap-table (probe A passes); stores land for the words we can see; the minimal
straight-line form (v4) is clean while the larger function (v3) is not.

Not established: everything else. The per-probe codegen does not yield a clean
discriminator — "variable `cincoffset` on a stack-derived capability" fits A, B, D
and E but is falsified by C, which has no `cincoffset` at all.

## Next (one board run, ZERO risk to the repro)

Raise `LADDER_DBG_SLOTS` to ~45 and re-run the **identical, unmodified**
`gp_diag3.dom`. This is a controller-only change, so the reproducing binary is
untouched, and it reveals `W1[4..7]` and the `W2` window in memory. That answers
the one remaining binary question — whether the unobserved words hold garbage (a
store-side fault after all) or the seeds (load-side, confirmed) — without
perturbing the only construct known to reproduce.

Only after that should v5 add per-element dumping *inside* v3's structure; note
that adding stores changes codegen, and v4 is a warning that perturbing the
structure can make the fault vanish.
