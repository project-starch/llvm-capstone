# r79-pmp-bare-2026-11-23

R-79 as one bare JTAG image, for any Capstone bitstream resident today.

R-79 (`docs/ref/ISSUES.md`): the PMP and CPMP checks judged each load or store with the NEIGHBOURING cycle's access
type, and reported the PMP fault's tval from the neighbouring cycle's address. So:
- a lone S/U store passed a PMP region without W;
- a load followed at once by an independent store was refused 7;
- a refused access reported the next access's address as tval.

The wiring is upstream CVA6's (the PMP extraction), on every bitstream since. It is fixed in RTL on capstone-ariane
b5b1-step3 (09b9cf9fd). Siblings: none.

**Scope:**
- The image runs plain RISC-V in S, bare (satp 0). No capability instruction runs there, so it runs unchanged on every
  Capstone tree.
- The Plain baseline (riscv-base 8db8f9cda) has its own pmp_data_if, with the same wiring. It is NOT covered by this
  PREREG's vectors: a run there needs its own simulation.

**Contents:**
- `tests/r79-pmp.S`: the arms. It derives from capstone-ariane b5b1-step3 csmint-pmp.S, without the CSMINT arms, and its
  header lists every register it writes.
- `build.sh`: rebuilds the image with the wrapper of `r61-stale-bare-2026-10-24` (`inc/`).
- `images/SHA256SUMS`: pins the image.
- `PREREG.md`: the pre-registered vectors, simulated from THIS image on the resident trees' RTL and on the fixed tree,
  and the grading rule.
- Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG` = the image, `SUP_REC_ADDR` from its `.rec`), the
  bare-boot path only.
