# b5b2n-bare-2026-11-25

The next B5b-2 bitstream on silicon: 8a, 8b, the CCSRRW deletion and step 9 (capstone-ariane b5b2-step9 904cd9c6b),
as bare JTAG images. Design, rulings and readings: llvm-capstone capstone/docs/plans/2026-11-24-b5b2-trap-route.md.

**Sibling:** `b5b2-bare-2026-11-24` is B5b-2's own kit (on silicon at N=2). This kit differs from it in three ways.
- Step 9 makes the custom-3 debug family illegal on this silicon (CAPCREATE, CAPTYPE, CAPNODE, CAPPERM, CAPBOUND,
  CAPPRINT). So every test mints through CSMINT (-DCAP_MINT_CSMINT), and the recorder and the UART report are plain M
  stores (inc/asm_insn.h; M is integer under Sublet). The recorder's base is a register the test never names.
- The kit adds b5b2-illegal-csup (step 8a) and b5b2-ctxport's five arms (step 8's port of the retired context tests).
  b5b2-illegal now carries step 9's deletion arms.
- Three slot-keyed tests (nsedge, sref, ctxport) are re-keyed for CSMINT's root slot and carry a key check. See
  PREREG.md, "the kit's three findings".

**Images:** 22 graded, sublet-life (the control) first, plus `b5b2-cptag-diag` (added after board 2's first run: the
diagnostic for cptag's reading 21, a separate image; PREREG.md's amendment). `build.sh` builds them with riscv64-unknown-elf-gcc and checks:
- no custom-3 encoding is in any image except b5b2-illegal's six deletion arms, which serve as the check's control;
- `out/skew/` holds the key check's deliberately skewed builds, for simulation only.

`images/SHA256SUMS` pins the images; `PREREG.md` holds the vectors, simulated from these images without the scaffold,
and the grading rule; `expected-vectors.json` holds the same for graders. Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py`.
