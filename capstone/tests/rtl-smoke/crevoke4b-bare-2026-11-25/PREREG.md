# B5b-2 4(b) on silicon: pre-registered 2026-11-25, before any board run

**The bitstream.** The full build of capstone-ariane b5b2-crevoke4b 88db40027 (4(b) parts 1 and 2, the store-queue drain, the residual-2 guard), sealed by the board lane as `…-pagedtbl-r76-r75d-b5b2n-crev4b_88db40027.bit`. Its seal sha256 and the full-build keys are graded in llvm-capstone capstone/docs/plans/2026-11-25-crevoke-4b-rtl.md before any flash.

This bitstream is the 4(b) candidate, NOT the final one: the semantics owner's ruling of 2026-11-25 keeps "final" open while the B5T mcf / M-16 wedge stands. This kit says nothing about that wedge.

**The control tree.** The same 17 images were also simulated on 904cd9c6b's core files: the bitstream on boards 2 and 3 today. Its vectors are given beside the candidate's. This allows two readings:
- a run on 904cd9c6b before the flash is the instrument control, if the board lane chooses to make one;
- a run after the flash that reads the 904cd9c6b column means the flash did not take.

**The session.** One board, flashed with the seal, its name read back after the power cycle. Then the 17 images in the table's order, the control first. The runner power-cycles before each image. Every image is expected to RETURN (print SR and its vector, then spin), so the ordering rule is met.

**Each vector** is the recorder's words at `board_rec`: SR, then the values at board_rec + 8·(1..SR), as the UART prints them.
- They were taken by simulating THESE images (`out/*.elf`, the same build as `images/*.bin`) through the Anvil prologue, with `S12_MEM_DELAY=12` and `+time_out=3000000`.
- The candidate column is capstone-ariane 438a9c70b, rtl_dirty=no.
- The control column is 438a9c70b with core/capstone_sublet_life.sv, ex_stage.sv, cva6.sv and the three wt_dcache files restored from 904cd9c6b (s2pre-wt `.sup/k4kit-base`). The D$ files differ only in sim-only trace points.
- rc=2 is the image's CAP_PASS spin running to the cycle limit after the report.
- The wrapper (`inc/`) is byte-identical to b5b2n-bare-2026-11-25's (plain-M recorder and report, CSMINT minting). That kit is on silicon at N=2 on 904cd9c6b: Tier 1 20/21 as pre-registered with the control exact; its one MISS, b5b2-cptag reading 21, is ruled an instrument artifact in its README.

**Classes.** Every position carries a class, given here and in `expected-vectors.json`:

| class | meaning | how graded |
|---|---|---|
| E | exact | equal on silicon |
| M | measured: cycles, timer-trap counts, latencies (timing-dependent) | recorded, plus the bands below |
| R | recorded: ids and checksums that depend on where an execution stopped | recorded |
| U | ungraded: sublet-life's pool slots 5-9, above `next` | DRAM is not zeroed (b5b2-bare-2026-11-24's amendment) |

**Grading.**
1. **VOID boot:** sublet-life's E positions are not exact, or any RUN line's image sha16 or rec differs from SHA256SUMS / `.rec`.
2. **VOID image:** its SR differs from the table's SR on both columns.
3. **VOID arm (88db40027 only; the interrupt arms: crevint-mt, -sh, -lat, every crevsv):** timer traps at the site (position 9, 1-based) read 0. The timer never landed inside the CREVOKE, so the arm tested nothing. On 904cd9c6b position 9 reads 0 by construction (the CREVOKE is not interruptible there), so the rule does not apply to that column.
4. **AS PREDICTED (candidate):** every E position equals the 88db40027 column.
5. **Not flashed:** every E position equals the 904cd9c6b column where the two columns differ.
6. **FAIL:** anything else.

**Bands on the M positions** (graded, written before any run):
- **crevint-lat.** Positions 13-20 (1-based) are lat[0..7] in mtime ticks, as signed 64-bit; a negative entry is a spurious second trap and is skipped. Positions 21-22 are the calibration's cycles and ticks.
  - Each latency after the first that is positive, converted to cycles as lat × cycles/ticks, must be at most 15,000. Sim reads 3,692 at memory delay 12.
  - The first latency (the trap landing in the first execution's forward scan, which is O(n)) is recorded, not bounded.
  - VOID if no positive latency follows the first.
- **sbdrain-cost / -cost0.** Position 2 is 16 rounds' cycles. Both are recorded, on whichever tree runs them.
  - Sim (88db40027 / 904cd9c6b): cost 0x11e6 / 0x11e6, cost0 0x1138 / 0x10fa.
  - The two trees differ in more than the drain (4(b)'s CREVOKE is in every round), so the drain's own cost is NOT separable in these images. The plan's in-tree reading (capstone/docs/plans/2026-11-25-crevoke-4b-rtl.md, the drain section): 16 rounds take 4,283 cycles against 904cd9c6b's 4,221, and 4,600 against 4,538 with four stores ahead of each round. That is +62 either way, so having stores ahead adds nothing measurable there.
- **Every other M position** is recorded.

**What silicon cannot show.** Whether a restart RESUMED or RESCANNED: the path is in the unit's simulation trail only (crevsv's header). On silicon the crevsv arms check that the answers and the checksum do not depend on the path.

## Identities and vectors

E positions are plain values; the others are written <class:value>. 1-based positions are used in the rules above.

| image | sha256/16 | board_rec | SR (88db40027) | vector on 88db40027 | SR (904cd9c6b) | vector on 904cd9c6b |
|---|---|---|---|---|---|---|
| sublet-life | ecbfe9171c0dbbbf | 80003200 | 35 | 0x0 0x0 0x0 0x19 0x0 0x0 0x19 0x0 0x0 0x0 0x19 0x19 0x2 0x2 0x8000000000010000 0x6 0x300000003 0x8 0x0 0x0 0x1000000000 0x100000000 0x1400000000 0x400000000 <U:0x0> <U:0x200000000> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> 0x6 | 35 | 0x0 0x0 0x0 0x19 0x0 0x0 0x19 0x0 0x0 0x0 0x19 0x19 0x2 0x2 0x8000000000010000 0x6 0x300000003 0x8 0x0 0x0 0x1000000000 0x100000000 0x1400000000 0x400000000 <U:0x0> <U:0x200000000> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> <U:0x0> 0x6 |
| b5b2-crevint | 831edd9db32bdcec | 8000c000 | 12 | 0x0 <M:0x373b> 0x80000007 0x800000cf 0x800000ce 0x19 0x0 0x0 0x0 0x0 0xe2b9b216cdcfbdf1 0x1 | 12 | 0x0 <M:0x347f> 0x80000007 0x80000008 0x80000009 0x19 0x0 0x0 0x0 0x0 0x3365444f5c1c0d1 0x1 |
| b5b2-crevint-mt | 57f8776d5bde1c53 | 8000c000 | 12 | 0x0 <M:0x3d5f> 0x80000007 0x800000cf 0x800000ce 0x19 0x0 <M:0x8> <M:0x8> 0x0 0xe2b9b216cdcfbdf1 0x1 | 12 | 0x0 <M:0x3936> 0x80000007 0x80000008 0x80000009 0x19 0x0 <M:0x4> <M:0x0> 0x0 0x3365444f5c1c0d1 0x1 |
| b5b2-crevint-sh | 49492413eb26e56b | 8000c000 | 12 | 0x0 <M:0x41a1> 0x80000007 <R:0x100000047> <R:0x800000cf> 0x19 0x19 <M:0x2> <M:0x2> <M:0x1> <R:0x9227b3d4c7c6bfb1> 0x2 | 12 | 0x19 <M:0x3abd> 0x80000007 <R:0x80000008> <R:0x80000009> 0x19 0x18 <M:0x2> <M:0x0> <M:0x0> <R:0x3365444f5c1c0d1> 0x3 |
| b5b2-crevint-lat | e1bb5e13692020a6 | 8000d000 | 22 | 0x0 <M:0x427c> 0x80000007 0x800000cf 0x800000ce 0x19 0x0 <M:0x6> <M:0x6> 0x0 0xe2b9b216cdcfb1f9 0x1 <M:0x549> <M:0xffffffffffffff80> <M:0x39b> <M:0xffffffffffffff88> <M:0x39b> <M:0xffffffffffffff88> <M:0x0> <M:0x0> <M:0x75a8> <M:0x1d6a> | 22 | 0x0 <M:0x3a07> 0x80000007 0x80000008 0x80000009 0x19 0x0 <M:0x2> <M:0x0> 0x0 0x3365444f5c1ccd9 0x1 <M:0xd33> <M:0xffffffffffffff80> <M:0x0> <M:0x0> <M:0x0> <M:0x0> <M:0x0> <M:0x0> <M:0x75a8> <M:0x1d6a> |
| b5b2-crevsv-none | 9ec34a43ab485667 | 8000e000 | 12 | 0x0 <M:0x417f> 0x80000007 0x800000cf 0x800000ce 0x19 0x0 <M:0x7> <M:0x7> <M:0x2> 0xe2b9b216cdcf85e1 0x1 | 12 | 0x0 <M:0x3b46> 0x80000007 0x80000008 0x80000009 0x19 0x0 <M:0x2> <M:0x0> <M:0x0> 0x3365444f5c1f8c1 0x1 |
| b5b2-crevsv-srev | 3dc03002dffd8131 | 8000e000 | 12 | 0x0 <M:0x47b3> 0x80000007 0x800000cf 0x800000ce 0x19 0x0 <M:0xc> <M:0xa> <M:0x3> 0xe2b9b216cdcf85e1 0x1 | 12 | 0x0 <M:0x3bb8> 0x80000007 0x80000008 0x80000009 0x19 0x0 <M:0x2> <M:0x0> <M:0x0> 0x3365444f5c1f8c1 0x1 |
| b5b2-crevsv-straw | 91816175b0159a79 | 8000e000 | 12 | 0x0 <M:0x478a> 0x80000007 0x800000cf 0x800000ce 0x19 0x0 <M:0xc> <M:0xc> <M:0x3> 0xe2b9b216cdcf85e1 0x1 | 12 | 0x0 <M:0x3ba2> 0x80000007 0x80000008 0x80000009 0x19 0x0 <M:0x2> <M:0x0> <M:0x0> 0x3365444f5c1f8c1 0x1 |
| b5b2-crevsv-derive | cc20f27e0600a4cc | 8000e000 | 12 | 0x0 <M:0x48ee> 0x80000007 <R:0x800000cf> <R:0x800000ce> 0x19 0x0 <M:0xc> <M:0xa> <M:0x3> <R:0xd443b2166d8185e1> 0x1 | 12 | 0x0 <M:0x3d8a> 0x80000009 <R:0x8000000a> <R:0x8000000b> 0x19 0x0 <M:0x4> <M:0x0> <M:0x0> <R:0xe3377a44a5c188c1> 0x1 |
| b5b2-crevsv-kill | a14a376f067acab1 | 8000e000 | 12 | 0x0 <M:0x459a> <R:0x80000008> <R:0x800000d0> <R:0x800000cf> 0x19 0x0 <M:0xc> <M:0xa> <M:0x3> <R:0x5323b3b7586ffcf6> 0x1 | 12 | 0x0 <M:0x3aee> <R:0x80000008> <R:0x80000009> <R:0x8000000a> 0x19 0x0 <M:0x2> <M:0x0> <M:0x0> <R:0xf3372c44d23f41be> 0x1 |
| b5b2-sbdrain-none | 7963832cdb6d9721 | 8000c000 | 4 | 0x1e 0x0 0x100 0x1 | 4 | 0x1e 0x0 0x100 0x1 |
| b5b2-sbdrain-fill | 2d919f9cd6d8784b | 8000c000 | 4 | 0x1e 0x0 0x100 0x1 | 4 | 0x1e 0x0 0x100 0x1 |
| b5b2-sbdrain-fence | fe825b7b8cf2f853 | 8000c000 | 4 | 0x1e 0x0 0x100 0x1 | 4 | 0x1e 0x0 0x100 0x1 |
| b5b2-sbdrain-mint | cb734bb8aefd8721 | 8000c000 | 4 | 0x1e 0x0 0x100 0x1 | 4 | 0x1e 0x0 0x100 0x1 |
| b5b2-sbdrain-sret | cf43854116560f2d | 8000c000 | 4 | 0x1 0x0 0x1 0x0 | 4 | 0x1 0x0 0x1 0x0 |
| b5b2-sbdrain-cost0 | e5e888ed0408fe28 | 8000c000 | 4 | 0x1e <M:0x1138> 0x0 0x1 | 4 | 0x1e <M:0x10fa> 0x0 0x1 |
| b5b2-sbdrain-cost | 09c0f1ae017cca15 | 8000c000 | 4 | 0x1e <M:0x11e6> 0x0 0x1 | 4 | 0x1e <M:0x11e6> 0x0 0x1 |

**Reading the base column.**
- **crevint and crevsv:** on 904cd9c6b, the ids are the list-order release, and timer traps at the site read 0 (see the crevint header's base readings).
- **crevint-sh:** on 904cd9c6b it reads crevoke's cause 0x19, a test artefact. The base CREVOKE completes, the interrupt lands at the next instruction, and the handler's CDERIVE from the dead o writes s1 before it is read. Its M trap count is 0x3.
- **sbdrain:** the base reads as the candidate on every E position. No exposure in five shapes at memory delay 12 (the plan's record).

**The aperture build.** The aperture build (capstone-ariane b5b2-wedgeap and its successor) changes no architectural value. These images are simulated on it separately, and its seal is graded against the 88db40027 column only after that simulation equals it.

## Addendum 2026-11-25: the final candidate a5c45a46d (R-80's fix), pre-registered before its seal

The same 17 images (byte-identical: SHA256SUMS unchanged) were simulated on capstone-ariane b5b2-ldcfix a5c45a46d, rtl_dirty=no (s2fix-wt `.sup/fin`), with the same settings as above. Their vectors are in `expected-vectors-a5c45a46d.json`, which has the same fields as `expected-vectors.json` plus `sim_rtl`.

**Every image reads exactly this kit's candidate column**, in every position including the M ones. R-80's pop changes no value of this kit.

On the a5c45a46d seal the rules above apply unchanged, with `expected-vectors-a5c45a46d.json`'s `vector` as the candidate column. As on 28fcbb646, a run on that seal cannot detect a missed flash from a 28fcbb646 board by this kit alone: those columns are equal. The flash is shown by the readback, and by b5b2-ldcfix-bare-2026-11-25 where the board is known to wedge without the fix.

## Addendum 2026-11-25: the final candidate b42c8c644 (R-80 + R-77's PRE fix), pre-registered before its seal

The same 17 images (byte-identical: SHA256SUMS unchanged) were simulated on capstone-ariane b5b2-prefix b42c8c644 (core = 7861abcfa), rtl_dirty=no (s2pre-wt `.sup/r77`), with this kit's settings. Their vectors are in `expected-vectors-b42c8c644.json`, which has the same fields as `expected-vectors.json` plus `sim_rtl`.

**Every image reads exactly this kit's `vector` column (the a5c45a46d / candidate column), in every position.** R-77's fix changes no value of this kit. On the b42c8c644 seal the rules above apply unchanged, with `expected-vectors-b42c8c644.json`'s `vector` as the column.
