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
