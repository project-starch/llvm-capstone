# RETRACTION + the sharpest result yet: the gp-captable fault is TWO faults

**Date:** 2026-07-26 · **Lane:** B · Board run: gp_diag3 at `LADDER_DBG_SLOTS=45`,
**identical unmodified dom** (sha `158760ea74175579`), controller-only rebuild.
Board left powered off + unlocked.

## RETRACTION FIRST

The immediately preceding note
(`25-07-2026_20-12-00_gp-diag-v3-shared-region-hypothesis-refuted.md`, "The
decisive datum: the STORES LANDED") and **commit `35951d5a`'s message** both claim:

> the stores land, so it is a **load-side** fault.

**That claim is wrong and is withdrawn.** It was drawn from a 33-slot run that
could only see `W1[0..3]`. The very caveat written next to it — that `W1[4..7]`
were unobserved — is exactly where the truth was hiding. With the window fully
visible, **`W1[7]` is corrupt in memory**. There *is* a store-side component.

The commit message is already pushed and cannot be edited; this note is the
correction of record. The lesson is the one the determinism control already
taught and I did not generalise: **do not promote a partial observation to a
conclusion.** "The 4 words I can see are fine" is not "the stores land."

## The run

```
gp_diag3  retval=590397633  oracle=1967218313  cycles=12047   NO

dbg0=65280      dbg1=2308472208 dbg2=2174254480 dbg3=2048
dbg4=134282752  dbg5=65280      dbg6=2308472208 dbg7=134218533
dbg8=12648430
dbg9..dbg28 = 0                        (unused gap res[12..32), as expected)
dbg29..dbg36 = 256 512 1024 2048 4096 8192 16384 **2174221968**   <- W1, host read
dbg37..dbg44 = 256 512 1024 2048 4096 8192 16384 32768            <- W2, host read
```

- **`W1[7]` = `res[39]` = 2174221968 = `0x8197FE90`** — a DRAM address
  (`0x80000000 + 0x197FE90`), **16-byte aligned**, where 32768 should be. The
  other seven words of `W1` are correct.
- **`W2` is completely intact** — and `W2` is written by **probe F's LOOP**
  (`for (i…) res[W2+i] = SEED(i)`). So loop stores through the shared-region
  capability land perfectly.

## A two-fault model explains 12/12 probe values across BOTH runs

**Fault 1 — one word of memory holds an address.** `res[39]` contains
`0x8197FE90` instead of its seed.

**Fault 2 — a spurious `+0x08000000` (2^27) in the loop accumulator.**

Together these predict every probe exactly, with no residue:

| probe | got | predicted | model |
|---|---:|---:|---|
| A loop, global array (gp table) | 65280 | 65280 | clean |
| B loop over `res[]` | 2308472208 | 2308472208 | memory + 2^27 |
| C straight-line read of `res[]` | 2174254480 | 2174254480 | **memory as-is — no anomaly** |
| D loop, constant index | 2048 | 2048 | clean |
| E loop over local stack array | 134282752 | 134282752 | sum − seed0 + 2^27 |
| F loop store + readback | 65280 | 65280 | clean |
| G walking pointer | 2308472208 | 2308472208 | memory + 2^27 |
| H nested byte-sum | 134218533 | 134218533 | bytes(0x8197FE90)=678, +127, +2^27 |

The same model, with the corrupt word inferred from probe C, reproduces the
**33-slot run's** numbers exactly as well (`W1[7]` = `0x819BFE90` there).

**`H` is the decisive probe.** It sums *bytes* (`(v >> 8b) & 0xff`), so its result
is bounded by 8·8·255 = 16320. It returned 134218533 = 2^27 + 805, where 805 is
the arithmetically correct byte-sum of the corrupt memory. A masked byte load
cannot produce 2^27, so **the 2^27 is injected into the accumulator, not into a
loaded value.**

## Consequence: the discarded hypothesis is RESURRECTED

Fault 2 appears in exactly **B, E, G, H** — loops that index a **stack- or
region-derived capability with a VARYING index** — and in none of A (global via
the `gp` cap-table), D (constant index), C (straight-line), or F (stores).

That is precisely the "**variable `cincoffset` on a stack-derived capability**"
hypothesis, which the previous note recorded as *"falsified by C, which has no
`cincoffset` at all."* **That falsification was an artifact.** C's wrongness was
never fault 2 — it was fault 1 leaking in through the shared memory word. Once
the two are separated, the discriminator is clean and fits all nine probes.

## The address is layout-dependent, not a fixed pattern

| run | controller | `W1[7]` |
|---|---|---|
| 33 slots | `LADDER_DBG_SLOTS=33` | `0x819BFE90` |
| 45 slots | `LADDER_DBG_SLOTS=45` | `0x8197FE90` |

Δ = `0x40000` (256 KiB); the **low 18 bits are identical** (`0x3FE90`). Changing
only the *controller* moved it, so the injected value is a live address that
tracks memory layout. Determinism still holds — it is per-configuration, and both
prior runs at 33 slots were bit-identical.

## Codegen: the store is CORRECT, so this is execution, not compilation

```
102fc: sd   a1, 0x130(a2)     ; res[38] = 16384
10300: ldc  a1, 0x0(a0)       ; reload the region capability
10304: lui  a0, 0x8           ; a0 = 32768
10308: sd   a0, 0x138(a1)     ; res[39] = 32768   <-- correct instruction
```

`0x138` = 312 = `res[39]`. The value and address are both right, so nothing
mis-compiled: either the store did not take effect, or something overwrote it.
Since the controller `memset`s the whole region to 0 before the share, a store
that simply vanished would leave **0**, not an address. **Something wrote a
capability-shaped value there.**

A 16-byte capability store landing at region offset `0x138` would put its address
word in `res[39]` and its metadata in `res[40]` — and `res[40]` is later
overwritten by probe F's loop, which is consistent with `W2` reading clean. The
domain does execute capability stores near there (`stc a1, -0xe0(s0)`), but `sp`
comes from `cscratch` = the **dom_data** region (`start-gp-captable-generic.S:43`),
a different capability from the shared region, so under enforced bounds it should
not be able to reach `res[]`. Whether those two regions actually overlap on this
silicon is now the open question — **this is a hypothesis, not a result.**

## Next

1. **Controller-only again (zero perturbation):** print the shared region's
   address alongside the dump, so `0x8197FE90` can be classified as inside the
   region, inside dom_data/stack, or neither. That single fact decides between
   "wild capability store" and "region/stack overlap".
2. Only then consider a v5 that reports `sp` and `&res[39]` from inside the
   domain — it adds stores, and v4 is the standing warning that perturbing the
   structure makes the fault vanish.
3. Fault 2 is now a clean, testable codegen statement (variable-index access off
   a non-`gp` capability) and can be attacked separately from fault 1.

Supersedes the "load-side" conclusion of
`25-07-2026_20-12-00_gp-diag-v3-shared-region-hypothesis-refuted.md`; the rest of
that note (the refutations of the shared-region and loop hypotheses, determinism,
the QEMU parity leg) stands.
