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
