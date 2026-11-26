# b5b2-nogrow-bare-2026-11-26

This kit reads NOGROW on Line B's final Full core, b42c8c644, on silicon. NOGROW is the growth rule that makes a full
table that may not grow answer a lifetime operation with the no-fault failure: no exception, an untagged result, and
nothing allocated. Cause 30 and growth are already on silicon (crevoke4b-bare-2026-11-25's b5b2-sbdrain images).
NOGROW was covered only at the unit level (sublet-life-diff.py's paged-full-nogrow) and, since 2026-11-26, at the core
level in simulation.

**Siblings:** `b5b2-r60r32-bare-2026-11-26`, `b5b2-misscost-bare-2026-11-26` and the other B5b-2 kits run on the same
seals. The wrapper (`inc/`) is theirs, byte-identical, and so is `sublet-life`'s image.

**Images, in this order** (every one returns):
- `sublet-life`: the boot control.
- `b5b2-nogrow-ctl` (-DNG_CTL): b5b2-sbdrain.S's full paged table with the growth word 0. S's CDERIVE raises 30, as
  b5b2-sbdrain-none reads on silicon.
- `b5b2-nogrow`: the same table with the growth word's bit 0 set (NOGROW). The same CDERIVE takes no trap, returns an
  untagged rd, and leaves the header's next unchanged.

The test is capstone-ariane b5b2-hangmon-c's `verif/tests/custom/capstone/b5b2-nogrow.S` (da427ee88), unchanged. Its
header holds the in-tree reading. `PREREG.md` holds the grading.
