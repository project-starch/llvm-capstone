# The 17-rung ladder on the Sublet bitstream, both halves, layouts K = 0..3: pre-registration (2026-10-16)

The board campaign's wave 1 item 1 (`docs/plans/2026-10-16-board-campaign-all.md`): the 15-kernel spatial cost, the
bare half paired with the capability half, over the layouts K = 0..3. It repeats `../ladder-revival-2026-09-22`
phase 10, "the layout-randomised table", on `caplifive_sublet_d93a8d66b.bit`. That phase took the medians of 4
layouts with bands, on `caplifive_m1_054cea69b`.

## K = 0 is already on record, and was not yet paired (`results/K0-s10.*`)

The Sublet acceptance's ladder boot (a7, s10, 2026-10-16 01:31-01:42) ran both halves at K = 0.
- The capability half: 17 of 17 rungs retval == oracle, instret identical to the FP respin's.
- The bare-metal baseline sweep ran right after it, every rung at 15/15 passes on min instret.
- The plan lists the capability half only, because the pair was never computed.

`../ladder-revival-2026-09-22/pair-halves.py` on the two, unchanged:
- **The control passes all three gates:** ctrsanity's instruction ratio is 1.00002, its baseline CPI is 1.2000, and
  its baseline is a floor (15/15). Its own cycle ratio is 1.0004.
- **17 admissible rows**, cycle ratios 0.981× (`beebs_cover`) to 2.227× (`beebs_insertsort`); `results/K0-s10.pair.txt`.

That is one draw of layout. Phase 10 showed per-kernel cycle ratios move with layout on this core family. So the
table to cite is the median over K = 0..3, which needs the three boots below.

## K = 1, 2, 3: what runs

Thursday's `ladder-k0.sh` (the Sublet acceptance's, which inherits the shell's CPU mask), with `K in 1 2 3` and its
output tags renamed:
- for each K, both halves are built with `-DLADDER_PAD=K` (file-scope `.p2align 6` + K nops) from `p9.spec`;
- then one capability boot (`run_ladder_perf_fpga.py`, `LADDER_ONE_BOOT=1`, distinct VAs);
- then one bare-metal sweep (`run_base_bare_fpga.py`);
- the classic line's shared tree, d04bd83 / df0e9a8 / 93440d5, as on Thursday.

## Pre-registered readings, per K

- **Capability half:** 17 of 17 rungs retval == oracle.
- **Instret is the layout-independent part.** Phase 10 read identical instret at every K for every rung. So each
  rung's capability instret equals Thursday's K = 0 value, and so does each bare rung's. Any rung whose instret moves
  is a finding, about the build rather than the layout.
- **Bare half:** every rung a floor (`clean >= 10`), retval equal to the capability half's.
- **pair-halves.py** passes its control gates at every K and prints all 17 rows.
- **Cycles are the reading**, reported per K and as the median over K = 0..3 with its range, as phase 10 did. Phase
  10's spread per kernel (on 054cea69b) is the expectation, not a gate: `beebs_cover` 0.95-1.14×, the control
  1.000× once its loop is aligned.

**Not established:** one boot per K, as phase 10. The capability half's CPI on this bitstream against the FP
respin's is a bridge only for K = 0: cycles within 0.74 %.
