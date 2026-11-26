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
