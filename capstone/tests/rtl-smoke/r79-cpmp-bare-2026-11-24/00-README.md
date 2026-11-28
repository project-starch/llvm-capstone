# r79-cpmp-bare-2026-11-24

R-79's CPMP side as one bare JTAG image, for the P1-line bitstream that carries R-79's fix.

R-79 (`docs/ref/ISSUES.md`): pmp_data_if judged each load or store with the NEIGHBOURING cycle's access type. The
CPMP's permission pick reads the same input (`cpmp_perm_ok`: perm[1] for a store, perm[2] for a read), so through a
CPMP entry:
- a legal read with a store translated in the next cycle was refused;
- a lone store passed an RO entry;
- a READ passed a W-only entry, and a legal store through it was refused as a read.

The PMP side is the sibling kit `r79-pmp-bare-2026-11-23` (fixed on silicon on the B5b-1 bitstream, which ties the
CPMP off). The fix is capstone-ariane 09b9cf9fd; on the P1 line it is p1-r79 7214834e3 (1c5d31474 plus the fix by
anchor, the changed lines diff-identical to 09b9cf9fd's).

**Scope:**
- The image runs CAPENTER and installs four NONLIN CPMP entries, so it needs a bitstream where CAPENTER is legal and
  the CPMP is live: the P1 line. It does NOT run on a Sublet build (d0afdb40f and its successors: CAPENTER illegal,
  CPMP tied off).
- Its graded vector is the FIXED row on the bitstream built from p1-r79. On a pre-fix P1-line resident the row
  depends on whether a following translation lands in the very next cycle, so nothing there is graded (see PREREG).

**Contents:**
- `tests/r79-cpmp.S`: the arms, byte-identical to capstone-ariane p1-r79 7bd984edc
  `verif/tests/custom/capstone/r79-cpmp.S`. Its header lists every register it writes.
- `build.sh`: rebuilds the image with the wrapper of `r79-pmp-bare-2026-11-23` (`inc/`, identical).
- `images/SHA256SUMS`: pins the image.
- `PREREG.md`: the pre-registered vector, simulated from THIS image on p1-r79's RTL, the information-only run on
  1c5d31474, and the grading rule.
- Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG` = the image, `SUP_REC_ADDR` from its `.rec`), the
  bare-boot path only.
