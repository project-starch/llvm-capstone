# b5b2-misscost-bare-2026-11-26

This kit measures the verdict-miss cost on silicon: the time half of the Line B record-cache decision (the semantics owner's rule 2, 2026-11-26). Events come from capstone/tools/record-cache-model/rcm-lb.c; time comes from a MEASURED per-miss cost, and this is its silicon reading. The cost is read from the same images on one boot: a matched pair inside each image.

**Siblings:** `b5b2-ldcfix-bare-2026-11-25` and the other B5b-2 kits run on the same seals. The wrapper (`inc/`) is theirs, byte-identical, and so is `sublet-life`'s image.

**Images, in this order** (every one returns):
- `sublet-life`: the control.
- `b5b2-srvcost`: eight passes of 32 LDC + 32 LD in protected U, with S acting between passes. P3 follows a namespace change (srevroot to tbl2 and back) and P7 an M round trip, so each re-probes 33 ids with the records warm.
- `b5b2-srvcost-cold` (-DSC_COLD): the same passes, plus the D$ evicted before P3 (256 KiB of loads in S, outside the timed window) and, as the control, before P4 alone. P3 then probes with its records cold.

The test is capstone-ariane 3ef6263fb's `verif/tests/custom/capstone/b5b2-srvcost.S`, unchanged. Its header holds the in-tree readings and the pre-registration. `PREREG.md` holds the grading.
