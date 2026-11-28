# b5b2-ldcfix-bare-2026-11-25

This kit runs R-80's fix on silicon: capstone-ariane b5b2-ldcfix a5c45a46d, in which a faulting LDC pops like a faulting plain load. It is the final B5b-2 candidate: 4(b) + the drain + the guard + the FLU rule + the wedge apertures + R-80. ISSUES R-80 and S-35 hold the mechanism. The record is llvm-capstone capstone/docs/plans/2026-11-25-crevoke-4b-rtl.md.

**Siblings:**
- `b5b2-wapflu-bare-2026-11-25` and `crevoke4b-bare-2026-11-25` run on the same seal.
- The wrapper (`inc/`) is theirs, byte-identical (b5b2n's).
- `sublet-life`'s image is byte-identical to theirs.

**Images, in this order:**
- `sublet-life`: the control.
- `b5b2-ldcexc2-ctl`: protected U; 16 + 1 stores through a capability to distinct lines, then at once an LDC through the valid base.
- `b5b2-ldcexc2`: the same, with the LDC through an untagged base (24 from the LSU's capability check). This is R-80's trigger.

The test is capstone-ariane a5c45a46d's `verif/tests/custom/capstone/b5b2-ldcexc2.S`, unchanged. It mints through CSMINT here, and no image holds a custom-3 encoding (checked by build.sh). `PREREG.md` holds the vectors and the grading, and `expected-vectors.json` carries the same for graders.
