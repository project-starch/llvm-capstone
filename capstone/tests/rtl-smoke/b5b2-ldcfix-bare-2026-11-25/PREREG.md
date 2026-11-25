# R-80's fix on silicon: pre-registered 2026-11-25, before any board run

**The bitstream.** The full build of capstone-ariane b5b2-ldcfix a5c45a46d (28fcbb646 + R-80's pop: `if (state_q == SEND_TAG_LDC) pop_ld_o = 1'b1;`). The board lane seals it as `…-crev4b-wap-flu-r80_a5c45a46d.bit`. Before any flash, its seal sha256 and full-build keys are graded in llvm-capstone capstone/docs/plans/2026-11-25-crevoke-4b-rtl.md.

**The control tree.** The same three images were simulated on 28fcbb646, the diagnostic seal, which has no R-80 fix. Its column is given beside the fixed one.

**Each vector** is the recorder's words at `board_rec`: SR, then the values at board_rec + 8·(1..SR), as the UART prints them.
- They were taken by simulating THESE images (`out/*.elf`, the build of `images/*.bin`) through the Anvil prologue, with `S12_MEM_DELAY=12` and `+time_out=3000000`.
- The fixed column ran on s2fix-wt at a5c45a46d, rtl_dirty=no. The control column ran on s2pre-wt at 28fcbb646, rtl_dirty=no.
- rc=2 is the image's CAP_PASS spin running to the cycle limit after the report.

| # | image | sha16 | rec | a5c45a46d (fixed) | 28fcbb646 (control tree) |
|---|---|---|---|---|---|
| 1 | sublet-life | ecbfe9171c0dbbbf | 0x80003200 | SR 35, the wapflu / crevoke4b kits' vector | SR 35, the same vector |
| 2 | b5b2-ldcexc2-ctl | b192317dad75e50c | 0x8000d000 | SR 2: 0x0 0x0 | SR 2: 0x0 0x0 |
| 3 | b5b2-ldcexc2 | 704f139900248efc | 0x8000d000 | SR 2: 0x18 0x1 | **no report**: the core wedges (25,373 instructions retired in sim, recorder never written) |

Every position is class E, except sublet-life's pool slots 5-9 (positions 25-34), which are U as in the sibling kits (DRAM is not zeroed).

`b5b2-ldcexc2`'s two words are the trap's mcause (0x18 = 24, the LSU capability check on an untagged base) and the trap count (1).

**The session.** One board with the seal resident, its name read back after the power cycle. The three images run in the table's order, the control first. The runner power-cycles before each image. `b5b2-ldcexc2` is LAST, because on any board without the fix it does not return.

**Grading on the fixed seal.**
1. **VOID boot:** sublet-life's E positions are not exact, or any RUN line's image sha16 or rec differs from SHA256SUMS / `.rec`.
2. **VOID arm:** `b5b2-ldcexc2-ctl` does not read 0x0 0x0. Without the control, a failure of the arm cannot be separated from a broken backlog skeleton.
3. **AS PREDICTED:** `b5b2-ldcexc2` reads SR 2, 0x18 0x1.
4. **FIX FAILS on silicon:** `b5b2-ldcexc2` prints no SR within the runner's timeout (a wedge), on a boot that is not VOID. This is NOT a VOID.
5. **FAIL:** anything else, such as a different cause or trap count, or a second trap.

**On a pre-fix seal** (28fcbb646 or older), if this kit is run there as the semantics owner may allocate:
- **REPRODUCED ON SILICON:** the arm prints no SR, after the control and the ctl arm read exact. That is R-80's wedge on silicon.
- **NOT REPRODUCED:** the arm reads 0x18 0x1. Silicon drained the write buffer fast enough that the LDC reached the commit head; the backlog depends on DRAM latency, which sim models as a fixed delay of 12. That is not evidence against R-80 (the sim reading stands) and it makes this kit non-discriminating on that board.
- **FAIL:** anything else.

**What silicon cannot show.** Which link starved: that is read in sim only (ISSUES R-80). A wedge on a pre-fix seal can be read on the LED apertures (bytes 192, 223, 209, 229, 195) against the pattern pre-registered for mcf (llvm-capstone land/rtl-crevoke4b c7877d008be7). In sim this arm read 192 0xd1, 223 0x08, 209 0x10 and 229 0x94.

## Addendum 2026-11-25: the final candidate b42c8c644 (R-80 + R-77's PRE fix), pre-registered before its seal

The same 3 images (byte-identical: SHA256SUMS unchanged) were simulated on capstone-ariane b5b2-prefix b42c8c644 (core = 7861abcfa), rtl_dirty=no (s2pre-wt `.sup/r77`), with this kit's settings. Their vectors are in `expected-vectors-b42c8c644.json`, which has the same fields as `expected-vectors.json` plus `sim_rtl`.

**Every image reads exactly this kit's `vector` column (the a5c45a46d / candidate column), in every position.** R-77's fix changes no value of this kit. On the b42c8c644 seal the rules above apply unchanged, with `expected-vectors-b42c8c644.json`'s `vector` as the column.

## Addendum 2026-11-26: the spatial-only core 14f2ef81a, pre-registered before its board session

**Role:** INFORMATIONAL in board 1's spatial-only kit boot (runtime's allocation, 2026-11-26): read and recorded, graded by the rules below, no decision rests on it.

**The seal:** caplifive_…-r77-spatial_14f2ef81a.bit, sha256 df9de4d49b21b475934e3a90ee81722ece98f6b1207c893bbc9a475e4c928be9 (graded SP-FULL PASS, llvm-capstone land/rtl-crevoke4b 42fa22984689). The resident name is read back and the seal is cited by its sha256.

**The vectors:** the same 3 images, byte-identical (images/ unchanged since this kit's pin; every `out/*.elf` objcopies to its `images/*.bin`; SHA256SUMS verifies), were simulated through the Anvil prologue at `S12_MEM_DELAY=12` on capstone-ariane b5b2-spatial 14f2ef81a (core 5ff493d0a: the lifetime unit replaced by `capstone_sublet_life_stub`), rtl_dirty=no, time_out 3,000,000 (s2fix-wt `.sup/spk`, PROVENANCE 2026-11-25T18:48:16Z). They are in `expected-vectors-14f2ef81a.json`, with the same fields as `expected-vectors.json` plus `sim_rtl`.

**Against the Full core's column** (b42c8c644's (`expected-vectors-b42c8c644.json`)): b5b2-ldcexc2, b5b2-ldcexc2-ctl read identically. The positions that move (class in brackets; u = ungraded):

| image | sha16 | rec | SR | positions that move, Full → spatial-only |
|---|---|---|---|---|
| sublet-life | ecbfe9171c0dbbbf | 0x80003200 | 35 | 4[E] 0x19 → 0x0; 7[E] 0x19 → 0x0; 11[E] 0x19 → 0x1d; 12[E] 0x19 → 0x0; 15[E] 0x8000000000010000 → 0x0; 16[E] 0x6 → 0x0; 17[E] 0x300000003 → 0x0; 18[E] 0x8 → 0x0; 21[E] 0x1000000000 → 0x0; 22[E] 0x100000000 → 0x0; 23[E] 0x1400000000 → 0x0; 24[E] 0x400000000 → 0x0; 26[U] 0x200000000 → 0x0; 35[E] 0x6 → 0x3 |

Each move is the stub's stated semantics (llvm-capstone capstone/docs/plans/2026-11-26-spatial-only-lineb.md, "The 57 Line B tests: what MUST change"): every lifetime op completes with no effect, so a stale access is not refused (25 → 0, or the next check's cause: 25 → 28 where the access is also out of bounds; 25 → 4 where it is also misaligned; a dead-parent CDERIVE goes on to its size check, 29, at sublet-life's position 11, the plan's in-tree reading; the trap counts fall by the 25s that became no fault); CREVOKE takes a few cycles and the run stays live; ids read the root they were derived from; the lifetime table is never written (sublet-life's table positions read 0).

**On the 14f2ef81a seal** the rules above apply unchanged, with `expected-vectors-14f2ef81a.json`'s `vector` as the column. sublet-life's VOID rule reads its 14f2ef81a vector. That vector differs from every Full column at the positions above, so a board left on a Full bitstream is caught by this kit, as well as by the readback.
