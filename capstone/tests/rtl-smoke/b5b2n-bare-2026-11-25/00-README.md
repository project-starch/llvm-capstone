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

## Silicon readings

- Board 2, booking 1493 (2026-11-24, 22:13-22:35Z, run-b2-20261124T2213).
  **904cd9c6b CONFIRMED ON SILICON, N=1 (board 2): Tier 1 20/21 as pre-registered, the control exact; b5b2-cptag
  reading 21 MISS, classified by the pre-registered diagnostic (70dbac168818) as an instrument artifact: H-fetch 6/6 on
  both seals, warm delta 0 against a measured miss of 43-47 cycles.** (The semantics owner's wording, 2026-11-25.)
  - The seal …-r75d-b5b2n_904cd9c6b.bit was flashed with its sha256 checked, and read back after the flash and after
    the loads.
  - Every RUN line carries its image's sha16 and rec.
  - No KEY-CHECK FAILURE: nsedge, sref, sref-gen and ctxport's five arms are all as predicted, the reached-F4 marker
    included.
  - b5b2-illegal reads step 9's deletion on silicon: sites 19-23 and 25 trap 2 / 1 / 1, count 0x19.
  - Graded twice: by the board lane's grade-b5b2nkit.py, and here by a separate script reading the json from git.
- **The one MISS, b5b2-cptag position 22 (reading 21, a cycle comparison), read 0.**
  - Repeats on board 2 read 1, 1, 0; board 3 under the B5b-2 seal (0c0458149) read 1, 1, 1, and 0 once more in the
    diagnostic boot. Over all loads: about 2 zeros in 8 on 904cd9c6b, about 2 in 7 on 0c0458149, so it predates this
    bitstream.
  - **The pre-registered diagnostic (PREREG amendment) reads H-fetch on all six loads**, boards 2 and 3 (diag-b2 /
    diag-b3-20261124T2253):
    - with the code warm the CPTAG pass never changed the load's cost (4 / 4, flag 1);
    - each board's own miss scale is 43-47 cycles;
    - the one zero in the diagnostic boot had both cold windows slow (18 / 25).
  - Classified as an instrument artifact of the cold fetch window, not an RTL reading.
- N=1 on the new bitstream. Board 3 holds the B5b-2 seal.

