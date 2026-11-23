# csmint-bare-2026-11-23

B5b-1 step 3 on silicon: CSMINT in hardware, with the Sublet regression as the control, as bare JTAG images for the
first B5b-1 bitstream (capstone-ariane b5b1-step3 d0afdb40f: steps 1b, 2a, 2b and 3, the CPMP tie-off and R-79's fix).

**CSMINT** rd, rs1 = slot, rs2 = desc mints a NONLIN capability from a six-word descriptor into one or two memory slots,
with one record in the lifetime table at depth 0 (capstone-qemu c1985bb529 helper_csmint). It runs as a sequencer in
the LSU. Its design, rulings and simulation readings are in llvm-capstone capstone/docs/plans/2026-11-22-b5b1-format.md
(the step 3 sections).

**Siblings:**
- `r79-pmp-bare-2026-11-23` is R-79's kit. Its "after" vector is this bitstream's, and its image runs here unchanged.
- `csmint-pmp` here carries R-79's arms too, plus CSMINT's probe against the R-only page.

**Images**, the tests unchanged from capstone-ariane d0afdb40f `verif/tests/custom/capstone/`:
- `sublet-life`: step 2a's Sublet lifetime arms on the physical pool. The known-good control, first in the boot.
- `csmint`: 18 arms in integer M:
  - valid mints, with one and with two slots;
  - every fault with its tval;
  - every value refusal;
  - the representability edges;
  - rs1 = sp;
  - the minted granule read back, used and revoked;
  - a dependent read of rd;
  - a descriptor holding a stored capability (read RAW);
  - the pool's header.
- `csmint-vm`: CSMINT in S under Sv39. It covers the descriptor's page fault, the slots' store probes on a read-only
  page and on a D-clear page, unmapped VAs, and SUM cleared.
- `csmint-vm-nc`: the same image built with a non-canonical descriptor and slot (-DCM_NONCANON).
- `csmint-pmp`: S bare under an R-only PMP page. It runs R-79's arms and CSMINT's probe (7 on this bitstream).

`build.sh` rebuilds them with the wrapper of `r79-pmp-bare-2026-11-23` (`inc/`), and `images/SHA256SUMS` pins them.
`PREREG.md` holds the vectors, simulated from THESE images on d0afdb40f's RTL, and the grading rule.
Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG`, `SUP_REC_ADDR` from each `.rec`), the bare-boot
path only.
