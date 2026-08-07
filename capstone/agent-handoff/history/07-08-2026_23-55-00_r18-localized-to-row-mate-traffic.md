# R-18 localized: the victim needs an RMW'd SCALAR IN ITS OWN 16-byte ROW

**Date:** 2026-08-07. **Status of the previous framing: RETRACTED.**

## What was refuted first

The session opened intending to separate REGION (stack vs globals) from ADDRESS-CAPABILITY
PROVENANCE (`cincoffsetimm` off a register-resident `s0` vs a capability `ldc`-ed out of memory).
An adversarial audit refuted the premise that those are the only two variables, and it was right.
Between the stack accumulator `qc` and the global accumulator `gcnt` in the *same* stage-32 binary
there are at least four more:

1. **Row contents.** `qc`'s 16-byte row also holds `k` and `p`, both read-modify-written in the
   loop. `gcnt`'s row has nothing else stored into it, ever. This is the one that mattered.
2. **Position relative to the tainting `ldc`.** The global arm was inserted into the slot the
   failing store used to occupy; `qc` was displaced five instructions later. Loop body 25 → 30
   instructions, cycles 44080 → 52922.
3. **Capability bounds.** `gcnt` is reached through a 0x40-length capability; `s0` covers the whole
   remaining stack.
4. **Cache set**, roughly 110 sets apart.

Also: **"stack vs globals" was never two regions.** `gen-gp-captable-glue.py:177-201` carves both
out of the same `dom_data` by `split` off `sp`.

And the instrument itself had a hole: stage 32 packs its return through `slliw`/`andi 0xffff`, so
the one damage mode seen in the high bits (`0x08000237`) would have been reported as **"correct"**
for the global. "Boot 68 showed the global accumulator is exact" is unsupported at full width.

## The experiment that localized it

Hold everything the audit named, and change only row membership. `FDREG_GAP` in stage 19 is a
padding array declared between `qc` and `p,k`, so it moves the row-mates without touching the loop.

| rung | GAP | frame | layout | qc | cycles | iters |
|---|---|---|---|---|---|---|
| `k800` | — | — | control | — | 4744 | — |
| `c8` | 0 | 0x50 | k@0x14 p@0x18 qc@0x1c — **one row** | **567** | 44105 | 575.9 |
| `rg16` | 16 | 0x60 | k@0x14 p@0x18 \| qc@0x2c **alone** | **576** | 44173 | 576.8 |
| `rg32` | 32 | 0x70 | k@0x14 p@0x18 \| qc@0x3c **alone** | **576** | 44187 | 577.0 |

One boot, control first, four distinct entry VAs, all four entered and returned.

What is held across the three fdreg arms:

* the inner loop is **instruction-for-instruction identical** — verified by disassembly, including
  the victim RMW at the same `cincoffsetimm a1, s0, -0x34` in every arm;
* the victim keeps row offset **12**, the property all 9 damaged builds share;
* `k` and `p` keep `0x14`/`0x18`;
* region, provenance and bounds are untouched;
* **the cycle counts show the same ~576 iterations in all three**, so the cured arms did not get
  the right answer by doing less work — they ran the identical workload and did not lose the
  increment.

`c8` is byte-identical to the frozen repro image (`9ecd8c6f9eb2b23d`) and returned 67699255 for
the eighth consecutive boot.

**Conclusion: the defect requires another read-modify-written scalar in the victim's own 16-byte
row. Region and provenance are not the variables.** This is consistent with the lead already
recorded in `ISSUES.md` — the `k = 0` re-initialisation at `0x14`, bank 0 lanes 4-7, whose
dual-bank splash target at the shift8 geometry is exactly `0x1c`.

### The confound this result carries, stated plainly

All three arms have **different frame sizes** (0x50 / 0x60 / 0x70), so every absolute address
moved. "No longer shares a row" is therefore not yet separated from "moved to a different absolute
address and cache set". Four geometric laws in this investigation have already died to exactly
that, so the follow-up below is designed to remove it rather than to argue it away.

## The follow-up, with frame size HELD CONSTANT

Stage 36 splits the padding into `FDREG_GAPP` (between `qc` and `p`) and `FDREG_GAPK` (between `p`
and `k`). Holding `GAPP + GAPK` constant gives arms with the **same frame, the same number of
prologue stores, and the victim at the same absolute offset** — differing only in how many RMW'd
scalars remain in its row.

| arm | GAPP/GAPK | frame | row 1 | row 2 | qc row-mates |
|---|---|---|---|---|---|
| `rg16` (== `rmA`) | 16/0 | 96 | k@+4 p@+8 | **qc@+12** | **0** |
| `rmB` | 0/16 | 96 | k@+4 | p@+8 **qc@+12** | **1 (p)** |
| `c8` | — | 80 | k@+4 p@+8 qc@+12 | — | 2 |

`rmA` compiled **byte-identical to `rg16`** (`6dfcca90fe83`), which is a useful consistency check
and the reason only three images are staged. `rg16` and `rmB` have byte-identical inner loops.

This discriminates the two live readings:

* under the **`k = 0` splash** lead, `rmB` should be CLEAN — `k@0x14` and `qc@0x2c` are in
  different rows there, so no splash is possible, and `p@0x28` → bank 0 lanes 0-3 is `0x20`, not
  the victim;
* under a generic **"any RMW row-mate suffices"** reading, `rmB` should be WRONG.

### Result: `rmB` is CLEAN

Second boot, same shape, control first, all four entered:

| rung | qc | cycles | iters | |
|---|---|---|---|---|
| `k800` | — | 4704 | — | control OK |
| `c8` | **567** | 44076 | 575.6 | anchor, damaged — 9th consecutive boot |
| `rg16` | 576 | 44185 | 577.0 | 0 row-mates — clean, reproduces the first boot |
| `rmB` | **576** | 44177 | 576.9 | 1 row-mate (`p`) — **clean** |

`rg16` and `rmB` are frame-matched (96), victim at the same absolute `0x2c`, byte-identical inner
loops. **`p` sharing the victim's row is not sufficient.** The row-mate that matters is `k` — the
scalar re-zeroed at the top of every outer pass — which is precisely what `ISSUES.md` predicted.

## The frame-matched positive control (`rmC`)

`p` being irrelevant leaves one hole: every arm that has FAILED so far (`c8`) has frame 0x50, and
every arm that has PASSED has frame 0x60 or 0x70. So "k in the row" is still confounded with
"frame 0x50". `FDREG_GAPT` closes it — a trailing pad declared BELOW `k`, which grows the frame
without separating any of the three counters:

| arm | frame | store | qc abs | row 2 contents | result |
|---|---|---|---|---|---|
| `rg16` | 96 | 0 | 0x2c | qc@+12 alone | 576 |
| `rmB` | 96 | 0 | 0x2c | p@+8 qc@+12 | 576 |
| `rmC` | 96 | 0 | 0x2c | **k@+4** p@+8 qc@+12 | *(pending)* |

Identical frame, identical store offset, identical absolute victim address, identical cache set.
If `rmC` returns 567 the frame-size confound is dead and the necessary condition is established as
"`k` is in the victim's row", with everything else held.

## Tooling repaired in the same session (commit `769a9a1cb9c7`)

Four checks written to prevent wrong verdicts were not running:

* **initramfs membership** — a `.dom` in `overlay/` but not in the packed cpio makes `lpc` exit
  non-zero, which the runner classifies as an R-16 entry stall and tells the operator to redraw an
  image that was never on the board. `gvf0`/`gvf6` were in exactly that state. New gate C14.
* **entry-stall streak (C11) was inert** — it tested `SHA6:` over the whole transcript, and the
  mandated control always emits `SHA6`, so it never fired once in its life.
* **wrong-oracle bug** — `run-ladder-qemu.sh` built the native oracle with a bare `cc -O0`, so a
  domain built at `-DFDREG_STAGE=N` was compared against a host binary at the default stage.
* **`.qemu-pass` had no writer** — the gate's own comment described a verify step that did not
  exist. `verify-and-stage-rung.sh` is now that step.

Stage 30 (hand-asm compiler-attribution test) is **retired**: it never compiled, and even fixed it
could not have tested the compiler, because the real failing build does `ldc a0 → lw a0 → sw a0`
through the *same* register and the hand-asm body used different ones. The question is answered by
disassembly instead — the emitted victim RMW is a plain `lw`/`addiw`/`sw` on a fixed stack offset,
so **the compiler is excluded by inspection**, which is a stronger test than hand asm because the
artifact is what actually executes.
