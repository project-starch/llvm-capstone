# wmem patch 0001, the Q-15 placement repair: pre-registration (2026-10-19, before any run of the repaired images)

Written on lane branch `q15-wmem` and pushed before the runs. The readings land in this folder beside it. Run roots are
on the build host under `/mnt/hdd/<user>/capstone/`. Kits:
- `qkit7` (capstone-qemu b290f5656600, module 5bfc38f808e1effd, exact bounds stripped);
- `qkit7-eb` (identical, exact bounds left on).

## Why

On the compact bounds format, CDERIVE refuses (29) a child it cannot hold exactly. That covers vp2's RTL at 8522d72fe
(`capstone_dyn_unit.anvil` :277-278, under the table namespace) and capstone-qemu without exact bounds
(`op_helper.c` :1240-1246). The vp2-nested-pools arm's uaf-jumbo control reads "none" with "fault in capstone_cap_derive"
on QEMU (wm1) and on silicon (tree A, `cause=29`, image offset 0x18a24, the same CDERIVE word). The jumbo block's life
derive, over 3 MiB plus headers, has a length that cannot be a multiple of its 4096 granule. Children of 16 KiB and up at
16-byte chunk offsets fail the same way.

## The repair (patch 0001 header, "Q-15")

- Children are padded at the tail only, to `capstone_repr_round`.
- From 16 KiB a child is placed on its granule inside a chunk served that much larger, with its offset stored below it.
- A realloc moves a placed object's bytes back for the allocator's copy and up again to the new placement.
- Jumbo blocks are malloc'd at `capstone_repr_round` and their lifetimes cover that length.
- The pad is counted, and the driver prints `Q15_PAD` after a completed case.

## Offline evidence

`capstone/tests/exact-bounds-iii/wmem-q15-check.py` runs on the patched header:
- **Real helpers:** compiled natively; 6,000 issues, 4,875 placed.
- **Failures:** 0 inexact children, 0 head pads, 0 children outside the served bytes, 0 bad stored offsets, 0 inexact jumbo lengths.
- **Oracle:** the QEMU compact-encoder model.
- **Controls:** the unrepaired child is refused 5,761 of 5,863 times, and the unrounded jumbo length 1,895 times.
- **Negatives:** a header without the helpers exits 1, and a mutant with placement disabled FAILs with 3,858 inexact children.

## Static checks on the built images

The tree was origin/dev a81857dcd6d0 plus this change, built with the t3-freeze-nb SDK, Debug, same flags.
- **OFF:** the cases-OFF and controls-OFF `.text` sections are identical to the images wm0 used (30/30).
- **ON:** all 30 ON images differ.
- **Warnings:** 59 against 59.
- **Symbols:** `wmem_sublet_q15_counts` is present in ON images only.

## Runs and predictions

Comparator `q15-cmp.py`: per cell, verdict, cause, symbol, controls, notes and infra.

**Probe**, controls 00 and 01 of the wm1 images on qkit7-eb, run once:
- uaf-jumbo completes its derive and faults 25 in `wm_probe`: the window test passes and the compact 29 is the
  representability test;
- or it faults 29 at code base + 0x18a24: the window test fails, and padding alone is not the repair.

**A**, the unrepaired wm1 images on qkit7-eb (the reference): controls fault/fault/fault, with uaf-jumbo
"fault in wm_probe"; cases CAUGHT 23, as 2026-10-11-virtual-23 on an older SDK.

**B**, the repaired images on qkit7-eb: equal to A cell by cell. A cell that is CAUGHT in A and MISSED here is a
PAD-MASKED candidate, reported as that verdict with its pad, never folded.

**C**, the same images on qkit7: equal to B cell by cell, with 0 `cause=29` at `capstone_cap_derive`.

**D**, the planted negative (controls built with `WMEM_SUBLET_Q15_PLANT_UNPADDED_JUMBO`) on qkit7: uaf-jumbo reads
"none" with "fault in capstone_cap_derive" (29) on every case, so NO-READING:control-failed 23.

**E**, the malloc arm (OFF, rebuilt, code identical) on qkit7: equal to t3-pred-wm0 cell by cell.
