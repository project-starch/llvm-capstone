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

### CORRECTION 2026-08-08 — the confound named below was the WRONG one

This section originally said: *"All three arms have different frame sizes (0x50 / 0x60 / 0x70), so
every absolute address moved."* **That is wrong, and an audit caught it.**

`s0` is the CALLER's `sp` and is identical across all arms (every caller prologue is byte-identical).
Measured off the artifacts, all four arms put the victim at **`s0-0x34`** — unchanged:

| arm | frame | qc | p | k | `stc` target |
|---|---|---|---|---|---|
| `c8` | 0x50 | **s0−0x34** | s0−0x38 | s0−0x3c | s0−0x50 |
| `rg16` | 0x60 | **s0−0x34** | s0−0x48 | s0−0x4c | s0−0x60 |
| `rmB` | 0x60 | **s0−0x34** | s0−0x38 | s0−0x4c | s0−0x60 |
| `rmC` | 0x60 | **s0−0x34** | s0−0x38 | s0−0x3c | s0−0x60 |

The `sp`-relative table used earlier (`qc` at 0x1c → 0x2c → 0x3c) is an artifact of **`sp` moving**,
not of the victim moving. The disassembly said so all along — `cincoffsetimm a1, s0, -0x34` appears
in every arm, and it was quoted in this very note as evidence that the loop was unchanged, while
the opposite conclusion was drawn two paragraphs later. The victim's absolute address, D-cache set
and bank-row are therefore **excluded**, not confounded.

**The real confound is the one that replaced it.** Growing the frame moved `k`, `p` AND the
capability store together. `k` leaving the victim's row is inseparable, in `rg16`/`rg32`/`rmB`, from
the store's row ceasing to be adjacent to the victim's row:

| arm | store's row | victim's row | relation | result |
|---|---|---|---|---|
| `c8` | [s0−0x50, s0−0x40) | [s0−0x40, s0−0x30) | **adjacent** | 567 |
| `rg16` | [s0−0x60, s0−0x50) | [s0−0x40, s0−0x30) | 2 rows | 576 |
| `rmB` | [s0−0x60, s0−0x50) | [s0−0x40, s0−0x30) | 2 rows | 576 |

Both readings fit every point measured so far. The prior "proximity to the capability store"
retraction does **not** cover this: it rested on `wp0` at 24 bytes, which kills a 12-byte threshold
and says nothing about 44 or 60 — and the corpus has a damaged victim at 40 bytes (`kb12`), so a
window in (40, 44] survives untouched.

Two further limits on what this boot showed:

* **"Region and provenance are not the variables" is NOT established by it.** Every arm here is a
  stack build; the boot contains no region contrast at all. Settling that needs a GLOBAL scalar
  *with* RMW row-mates shown damaged, and no such build exists.
* **Row occupancy is not SUFFICIENT.** In `c8` itself `p` sits at `s0-0x38` — upper half, offset 8,
  two RMW row-mates — and is exact (`p=64` decodes out of 67699255). The rule cannot say which of
  two upper-half row-mates gets hit.

Also: `rg16` is **byte-identical to `t16`**, already boarded at boot 57 with the same result, and
`rg32`'s layout matches `gp32` from boot 53/54. Those two arms re-measured known points; the new
information in this session is `rmB` and `rmC`, not the gap sweep.

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

### Result: `rmC` is DAMAGED (567) — and it kills the store-adjacency alternative

| rung | qc | cycles | iters | |
|---|---|---|---|---|
| `k800` | — | 4799 | — | control OK |
| `c8` | **567** | 44077 | 575.6 | anchor — 10th consecutive boot |
| `rmB` | 576 | 44163 | 576.7 | `k` out of the row — clean |
| `rmC` | **567** | 44087 | 575.7 | `k` back in the row — **damaged** |

`rmB` and `rmC` differ in **one thing**:

| | frame | qc | p | k | `stc` target |
|---|---|---|---|---|---|
| `rmB` | 0x60 | s0−0x34 | s0−0x38 | **s0−0x4c** | s0−0x60 |
| `rmC` | 0x60 | s0−0x34 | s0−0x38 | **s0−0x3c** | s0−0x60 |

Same frame, same victim address, same `p`, **same capability-store position two rows away in both**.
`k` moves 16 bytes and the result flips 576 ↔ 567.

That refutes the store-row-adjacency alternative outright: `rmC` has the store two rows from the
victim, exactly as the clean arms do, and is damaged anyway. **The variable is `k`'s position.**

## The geometry, stated precisely

"Shares a row" is too loose — `rg16` has `p` at row offset 8 (upper half) sharing a row with `k`
and `p` is exact. Evaluating the sharper predicate over the whole corpus (`fit-victim-rules.py`
dataset plus this session's points):

> **A victim in bank 1 at byte lanes L is zeroed when another read-modify-written scalar occupies
> bank 0 at the SAME lanes L in the same 16-byte row** — i.e. an RMW slot at `victim_offset − 8`.

| | fit |
|---|---|
| clean builds (`c0`, `rs0`, `t16`/`rg16`, `bs16`, `nr16`, `rmB`, `rg32`) | **7/7 — no false positives** |
| lost-increment builds (`c8`, `rs8`, `dp0`, `kb12`, `rmC`) | **5/5** |
| `rs4` (−72), `ka0` (−558) | **not explained** |
| `c4`, `t12`, `t0b` (+333/+330/+333) | the separately documented **extra-iteration** fault, not this one |

So the predicate is **SUFFICIENT and not necessary**: every build carrying the bank-0 twin is
damaged and no clean build carries one, but two lost-increment builds are damaged without it. It
is not the whole of R-18 and is not presented as such.

This is exactly the mechanism `ISSUES.md` named as the leading untested lead — the `k = 0`
re-initialisation at bank 0 lanes 4-7, splashing into bank 1 lanes 4-7. It is now tested, by a
controlled single-variable pair rather than by a fit.

### What is still NOT established

* **Region and provenance remain untested.** Every arm in these three boots is a stack build; there
  is no region contrast anywhere in them. Settling that needs a GLOBAL scalar carrying the bank-0
  twin in its row, shown damaged. That build does not exist yet, and it is the obvious next one.
* `rmB`/`rmC` are N=1 each; the pair is one controlled comparison, not a replication.
* Two lost-increment builds (`rs4`, `ka0`) still have no explanation.

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
