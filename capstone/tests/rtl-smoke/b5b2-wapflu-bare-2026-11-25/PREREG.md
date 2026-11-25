# B5b-2 combined build on silicon: b5b2-untagged, pre-registered 2026-11-25, before any board run

**The bitstream.** The full build of capstone-ariane b5b2-wedgeap-flu 28fcbb646, as the board lane seals it (`…-b5b2n-crev4b-wap-flu_28fcbb646.bit`). It carries 4(b), the store-queue drain, the residual-2 guard, the B5T mcf / M-16 wedge apertures, and the FLU's untagged-x[rs1] rule in every mode. Its sha256 and full-build keys are graded in llvm-capstone capstone/docs/plans/2026-11-25-crevoke-4b-rtl.md before any flash.

It is diagnostic plus 4(b), not final, until the wedge is resolved (the semantics owner's ruling). The crevoke4b-bare-2026-11-25 kit runs on it too. Its 17 vectors read identically in sim on 47d749153, and are to be confirmed on 28fcbb646 before that seal is graded with them.

**The control tree** is 88db40027's seal, resident on board 2 today. Its column was simulated on 47d749153, the apertures without the FLU rule. That build's architectural values equal 88db40027's: the 17 crevoke4b vectors are identical on both.
- Run on 88db40027 BEFORE the flash, b5b2-untagged is the instrument control: the old 24s.
- Run AFTER the flash, it reads the rule.

**Each vector** is the recorder's words at `board_rec`: SR, then the values at board_rec + 8·(1..SR).
- They were taken by simulating THESE images (`out/*.elf`) through the Anvil prologue at memory delay 12, +time_out=3000000, rtl_dirty=no on both trees (28fcbb646 and 47d749153).
- The images' vectors equal the in-tree arms' on both trees, all nineteen positions.
- rc=2 is the CAP_PASS spin.

**b5b2-untagged** (capstone-ariane 28fcbb646, unchanged; header has the arms):
- positions 1-8: protected U;
- positions 9-16: S;
- positions 17-18: plain M's CDERIVE_S inexact child (29) and its exact control (0);
- position 19: the trap count.

On the combined build, U and S read the integer results (NULL + 0x40, an integer + 0x10, SCC to 0x1234), a 24 at the result's first use in U, and 24 for a tagged rs2. On 88db40027 every one of those operations raises 24 and the sentinels stay.

**Grading.**
1. **VOID boot:** sublet-life's E positions are not exact, or an image's sha16 or rec differs.
2. **AS PREDICTED:** every E position equals the 28fcbb646 column.
3. **Not flashed:** b5b2-untagged equals the 88db40027 column.
4. **FAIL:** anything else.

sublet-life's positions 25-34 are U (ungraded: DRAM above `next`), as in the earlier kits. Every b5b2-untagged position is E.

## Identities and vectors

| image | sha256/16 | board_rec | SR | vector on 28fcbb646 | vector on 88db40027 (sim 47d749153) |
|---|---|---|---|---|---|
| sublet-life | ecbfe9171c0dbbbf | 0x80003200 | 35 | 0x0 0x0 0x0 0x19 0x0 0x0 0x19 0x0 0x0 0x0 0x19 0x19 0x2 0x2 0x8000000000010000 0x6 0x300000003 0x8 0x0 0x0 0x1000000000 0x100000000 0x1400000000 0x400000000 <U:0x0> <U:0x200000000> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> 0x6 | 0x0 0x0 0x0 0x19 0x0 0x0 0x19 0x0 0x0 0x0 0x19 0x19 0x2 0x2 0x8000000000010000 0x6 0x300000003 0x8 0x0 0x0 0x1000000000 0x100000000 0x1400000000 0x400000000 <U:0x0> <U:0x200000000> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> 0x6 |
| b5b2-untagged | 7393852dbe0265f0 | 0x8000d000 | 19 | 0x0 0x40 0x0 0x10 0x0 0x1234 0x18 0x18 0x0 0x40 0x0 0x10 0x0 0x1234 0x18 0x0 0x1d 0x0 0x4 | 0x18 0x5a5a 0x18 0x0 0x18 0x5a5a 0x18 0x18 0x18 0x5a5a 0x18 0x0 0x18 0x5a5a 0x18 0x0 0x1d 0x0 0xa |

## Addendum 2026-11-25: the final candidate a5c45a46d (R-80's fix), pre-registered before its seal

The same 2 images (byte-identical: SHA256SUMS unchanged) were simulated on capstone-ariane b5b2-ldcfix a5c45a46d, rtl_dirty=no (s2fix-wt `.sup/fin`), with the same settings as above. Their vectors are in `expected-vectors-a5c45a46d.json`, which has the same fields as `expected-vectors.json` plus `sim_rtl`.

**Every image reads exactly this kit's candidate column**, in every position including the M ones. R-80's pop changes no value of this kit.

On the a5c45a46d seal the rules above apply unchanged, with `expected-vectors-a5c45a46d.json`'s `vector` as the candidate column. As on 28fcbb646, a run on that seal cannot detect a missed flash from a 28fcbb646 board by this kit alone: those columns are equal. The flash is shown by the readback, and by b5b2-ldcfix-bare-2026-11-25 where the board is known to wedge without the fix.

## Addendum 2026-11-25: the final candidate b42c8c644 (R-80 + R-77's PRE fix), pre-registered before its seal

The same 2 images (byte-identical: SHA256SUMS unchanged) were simulated on capstone-ariane b5b2-prefix b42c8c644 (core = 7861abcfa), rtl_dirty=no (s2pre-wt `.sup/r77`), with this kit's settings. Their vectors are in `expected-vectors-b42c8c644.json`, which has the same fields as `expected-vectors.json` plus `sim_rtl`.

**Every image reads exactly this kit's `vector` column (the a5c45a46d / candidate column), in every position.** R-77's fix changes no value of this kit. On the b42c8c644 seal the rules above apply unchanged, with `expected-vectors-b42c8c644.json`'s `vector` as the column.

## Addendum 2026-11-26: the spatial-only core 14f2ef81a, pre-registered before its board session

**Role:** INFORMATIONAL in board 1's spatial-only kit boot (runtime's allocation, 2026-11-26): read and recorded, graded by the rules below, no decision rests on it.

**The seal:** caplifive_…-r77-spatial_14f2ef81a.bit, sha256 df9de4d49b21b475934e3a90ee81722ece98f6b1207c893bbc9a475e4c928be9 (graded SP-FULL PASS, llvm-capstone land/rtl-crevoke4b 42fa22984689). The resident name is read back and the seal is cited by its sha256.

**The vectors:** the same 2 images, byte-identical (images/ unchanged since this kit's pin; every `out/*.elf` objcopies to its `images/*.bin`; SHA256SUMS verifies), were simulated through the Anvil prologue at `S12_MEM_DELAY=12` on capstone-ariane b5b2-spatial 14f2ef81a (core 5ff493d0a: the lifetime unit replaced by `capstone_sublet_life_stub`), rtl_dirty=no, time_out 3,000,000 (s2fix-wt `.sup/spk`, PROVENANCE 2026-11-25T18:48:16Z). They are in `expected-vectors-14f2ef81a.json`, with the same fields as `expected-vectors.json` plus `sim_rtl`.

**Against the Full core's column** (b42c8c644's (`expected-vectors-b42c8c644.json`)): b5b2-untagged reads identically. The positions that move (class in brackets; u = ungraded):

| image | sha16 | rec | SR | positions that move, Full → spatial-only |
|---|---|---|---|---|
| sublet-life | ecbfe9171c0dbbbf | 0x80003200 | 35 | 4[E] 0x19 → 0x0; 7[E] 0x19 → 0x0; 11[E] 0x19 → 0x1d; 12[E] 0x19 → 0x0; 15[E] 0x8000000000010000 → 0x0; 16[E] 0x6 → 0x0; 17[E] 0x300000003 → 0x0; 18[E] 0x8 → 0x0; 21[E] 0x1000000000 → 0x0; 22[E] 0x100000000 → 0x0; 23[E] 0x1400000000 → 0x0; 24[E] 0x400000000 → 0x0; 26[U] 0x200000000 → 0x0; 35[E] 0x6 → 0x3 |

Each move is the stub's stated semantics (llvm-capstone capstone/docs/plans/2026-11-26-spatial-only-lineb.md, "The 57 Line B tests: what MUST change"): every lifetime op completes with no effect, so a stale access is not refused (25 → 0, or the next check's cause: 25 → 28 where the access is also out of bounds; 25 → 4 where it is also misaligned; a dead-parent CDERIVE goes on to its size check, 29, at sublet-life's position 11, the plan's in-tree reading; the trap counts fall by the 25s that became no fault); CREVOKE takes a few cycles and the run stays live; ids read the root they were derived from; the lifetime table is never written (sublet-life's table positions read 0).

**On the 14f2ef81a seal** the rules above apply unchanged, with `expected-vectors-14f2ef81a.json`'s `vector` as the column. sublet-life's VOID rule reads its 14f2ef81a vector. That vector differs from every Full column at the positions above, so a board left on a Full bitstream is caught by this kit, as well as by the readback.
