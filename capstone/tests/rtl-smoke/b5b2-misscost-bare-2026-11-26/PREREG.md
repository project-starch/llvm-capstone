# The verdict-miss cost on silicon: pre-registered 2026-11-26, before any board run

**The seal.** Any B5b-2 seal with R-80's pop. This means the final candidate b42c8c644 (sha256 6a09f34d…190a), or a5c45a46d, whose replay path is the same. The resident name is read back, and the seal is cited by its sha256.

**Each vector** is the recorder's words at `board_rec`: SR, then the values, as the UART prints them.
- They were taken by simulating THESE images through the Anvil prologue at `S12_MEM_DELAY=12`, on capstone-ariane 3ef6263fb (core = b42c8c644), rtl_dirty=no (s2pre-wt `.sup/mck`).
- `expected-vectors.json` carries them, with a class per position.

| # | image | sha16 | rec | sim vector (b42c8c644) |
|---|---|---|---|---|
| 1 | sublet-life | ecbfe9171c0dbbbf | 0x80003200 | SR 35, the sibling kits' vector |
| 2 | b5b2-srvcost | e3ab19824ef4be45 | 0x8000f000 | D1-D4 0x52f 0xdb2 0xde2 0xdb2, P1-P8 0xc71 0x83 0x309 0x83 0x83 0x83 0x25a 0x83, count 0x8 |
| 3 | b5b2-srvcost-cold | 143742c6d7266f98 | 0x8000f000 | D1-D4 the same, P1-P8 0xc71 0x83 0x17c7 0xda2 0x83 0x83 0x9b9 0xb8, count 0x8 |

**Classes.** In images 2 and 3, positions 1-12 are M (cycles, measured) and position 13 is E. The count is 8 by the test's known slip: s_done leaves 8 in the counter. sublet-life's classes are the siblings'.

**The readings it delivers** (the decision's inputs, recorded, not graded):
- warm per-miss cost: (P3 - P2) / 33 and (P7 - P2) / 33 of image 2. Sim: 19.6 and 14.3 cycles;
- cold per-miss cost: (P3 - P4) / 33 of image 3. Sim: 78.7 cycles;
- cold, independently: (P7 - P2) / 33 of image 3, where P7's probes read records evicted before P4. Sim: 71.5 cycles;
- the in-tree arm (not the kit's layout) read 14.4 warm and 70.0 cold. The kit's P3 differs from its P7 (19.6 against 14.3), which the in-tree arm does not show. That is a layout effect of the bare wrapper, recorded and not traced.

**Grading** (structure only; the costs are measurements):
1. **VOID boot:** sublet-life's E positions are not exact, or any RUN line's image sha16 or rec differs from SHA256SUMS / `.rec`.
2. **VOID image:** its SR is not 13, or position 13 is not 0x8.
3. **AS PREDICTED:**
   - image 2: P3 > P2 and P7 > P2; P4, P5, P6 and P8 each within 20 % of P2;
   - image 3: P4 > P2, P3 > P4 and P7 > P2.
   Without these the subtraction measures nothing.
4. **FAIL:** anything else, reported with its values.

**Silicon cost** is the cycles of rdcycle at the core clock.

**The decision rule** (the semantics owner's, 2026-11-26, written before any board run):
- It uses the SILICON readings, not the simulation's.
- For each program, misses (rcm-lb's LB probes) times the per-miss cost, as a share of the program's cycles, is compared with the 1.0 % threshold.
- Where two readings of one quantity disagree (warm: P3 against P7 of image 2; cold: P3 against P7 of image 3), the build goes ahead only if the threshold is met with BOTH. A pair that straddles the threshold makes the result UNDECIDED, and the next step is to trace the layout effect. The larger reading is never chosen, because that choice favours the cache.

## Addendum 2026-11-26: the spatial-only core 14f2ef81a, pre-registered before its board session

**Role:** INFORMATIONAL, NOT GRADED on this core (runtime's ruling, 2026-11-26): the stub makes no verdict misses, so the decision rule's subtractions measure nothing; the raw counts are read as the zero-miss confirmation.

**The seal:** caplifive_…-r77-spatial_14f2ef81a.bit, sha256 df9de4d49b21b475934e3a90ee81722ece98f6b1207c893bbc9a475e4c928be9 (graded SP-FULL PASS, llvm-capstone land/rtl-crevoke4b 42fa22984689). The resident name is read back and the seal is cited by its sha256.

**The vectors:** the same 3 images, byte-identical (images/ unchanged since this kit's pin; every `out/*.elf` objcopies to its `images/*.bin`; SHA256SUMS verifies), were simulated through the Anvil prologue at `S12_MEM_DELAY=12` on capstone-ariane b5b2-spatial 14f2ef81a (core 5ff493d0a: the lifetime unit replaced by `capstone_sublet_life_stub`), rtl_dirty=no, time_out 3,000,000 (s2fix-wt `.sup/spk`, PROVENANCE 2026-11-25T18:48:16Z). They are in `expected-vectors-14f2ef81a.json`, with the same fields as `expected-vectors.json` plus `sim_rtl`.

**Against the Full core's column** (b42c8c644's (this kit's `expected-vectors.json`, simulated on 3ef6263fb, core = b42c8c644)): no image read identically. The positions that move (class in brackets; u = ungraded):

| image | sha16 | rec | SR | positions that move, Full → spatial-only |
|---|---|---|---|---|
| b5b2-srvcost | e3ab19824ef4be45 | 0x8000f000 | 13 | 1[M] 0x52f → 0x379; 5[M] 0xc71 → 0x593; 7[M] 0x309 → 0x83; 11[M] 0x25a → 0x83 |
| b5b2-srvcost-cold | 143742c6d7266f98 | 0x8000f000 | 13 | 1[M] 0x52f → 0x379; 5[M] 0xc71 → 0x593; 7[M] 0x17c7 → 0xee2; 11[M] 0x9b9 → 0x83; 12[M] 0xb8 → 0x83 |
| sublet-life | ecbfe9171c0dbbbf | 0x80003200 | 35 | 4[E] 0x19 → 0x0; 7[E] 0x19 → 0x0; 11[E] 0x19 → 0x1d; 12[E] 0x19 → 0x0; 15[E] 0x8000000000010000 → 0x0; 16[E] 0x6 → 0x0; 17[E] 0x300000003 → 0x0; 18[E] 0x8 → 0x0; 21[E] 0x1000000000 → 0x0; 22[E] 0x100000000 → 0x0; 23[E] 0x1400000000 → 0x0; 24[E] 0x400000000 → 0x0; 26[U] 0x200000000 → 0x0; 35[E] 0x6 → 0x3 |

What the moves are: no verdict misses. In b5b2-srvcost P3 = P2 = P7 = 0x83: the probes that miss on Full hit on this core, which has no lifetime unit to probe (the stub answers every liveness question live at once). D1 0x52f → 0x379 has the shape the plan traced for the in-tree D1 (on Full the setup's record writes are still draining when D1 begins). In b5b2-srvcost-cold P4 is unchanged (0xda2) and P3 reads 0xee2: (P3 - P4) / 33 = 9.7 cycles, against 78.7 on Full. That residual is the cold image's own data refill and not a verdict cost. It is recorded, not traced. sublet-life moves as in the sibling kits.

**On the 14f2ef81a seal:** rules 1 (VOID boot, with sublet-life's 14f2ef81a vector) and 2 (VOID image: SR 13, position 13 = 0x8) apply. Rule 3 (AS PREDICTED) does NOT apply: it requires P3 > P2, which a core with no verdict miss cannot meet. Rule 4 is not graded. A reading is compared with `expected-vectors-14f2ef81a.json`: E positions exactly, M positions reported. The decision rule above is not evaluated on this core.
