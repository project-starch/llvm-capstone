# mruby virtual-nested-pools after patch 0011 (Q-15 placement) -- pre-registration, 2026-10-21

Written and pushed before any run of the repaired image. The readings land beside it in `RESULT.md`.

## Why

The vp2 corpus pass's compact-bounds references found the Sublet image (`25e86274ce192288`, capi-11 `9935a60f7c437031`)
unable to start. Every execution refused with cause 29 at a CDERIVE in `mrb_gc_init`: on the tB-mimic kit, 24 of 24
(23 at image offset 0x5154c in the mruby image, 1 at 0x4f848 in case 11's capi-11 image). On qkit7-nf2 the smoke
control fails the same way. Patch 0011 aligns `page->objects` to its bounds granule, so the page lifetime is exact.

That control reading was taken BEFORE this file was written; it is the observation that motivated the fix, not a
prediction.

## Images

- Null control: the stock images must be byte-identical to the corpus pass's, nb `3a198692aff951b8`, cw
  `8c7cd27614353067` (capi-11 nb `6d8f003b0512ef96`, cw `a6d22b0dd2627d55`). 0011 changes nothing without
  `MRB_CAPSTONE_SUBLET`. If they differ, the build moved and nothing below is attributable to 0011.
- The repaired Sublet images differ from `25e86274` and `9935a60f`. `identity-q15.py` exits 0 on the twin pairs.

## Step 1 -- exact bounds (qkit7-eb), the matched pair

There are two arms on qkit7-eb: the unrepaired cw Sublet image, then the repaired one. Both are compared, cell by cell
on (outcome, cause, function), with `results/2026-10-11-virtual/virtual-nested-pools/matrix.tsv`, and with each other.

- Unrepaired: smoke ok, read rc 0, and 23/23 cells equal the record.
- Repaired: smoke ok, read rc 0, and 23/23 cells equal the record and the unrepaired arm.
- Cause-29 refusals (`count29.py`): 0 in both arms.

The attributing comparison is repaired vs unrepaired on the same kit, which is one variable: patch 0011. The record
was built with a different SDK, so a record difference shared by both arms belongs to the build, not to 0011.

## Step 2 -- compact bounds: tB-mimic (the board's reference kit), then qkit7-nf2 (plain)

- Repaired: control smoke ok, read rc 0.
- Cause-29 refusals: **0** over smoke + 23 cases, on both kits.
  - The control is the unrepaired image's 24/24 on tB-mimic, read with the same counter.
  - The counter was checked first: 24 on the unrepaired capture, 0 on mruby-stock's, and "no data" exits 2.
- Cells: 23/23 equal step 1's repaired eb reading, and tB-mimic equals nf2.

The one admitted exception is a case that is CAUGHT under exact bounds and MISSED or NO-READING under compact bounds.
It counts only if it is shown pad-masked: its first out-of-bounds offset must be smaller than the padding the rounded
bounds add (`first_oob_offset < pad_bytes`). None is predicted.

## What would refute it

- Any cause-29 refusal on the repaired image: the placement is incomplete. For example, a region page from
  `mrb_gc_add_region`, or an allocator that ignores the alignment; the abort message in `init_heap_page` would say so.
- A cell that moves between the unrepaired and repaired arms on qkit7-eb. 0011 would then have changed behaviour
  beyond placement: the page is 48 bytes larger and objects[] sits 48 bytes later.
