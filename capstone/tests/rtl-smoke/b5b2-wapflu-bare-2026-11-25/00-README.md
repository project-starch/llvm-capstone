# b5b2-wapflu-bare-2026-11-25

This kit runs the combined B5b-2 build on silicon: capstone-ariane b5b2-wedgeap-flu 28fcbb646, which is 4(b) + the drain + the guard + the wedge apertures + the FLU's CHERI rule on an untagged x[rs1] in every mode. Its one test is b5b2-untagged as a bare JTAG image. The record is in llvm-capstone capstone/docs/plans/2026-11-25-crevoke-4b-rtl.md.

**Siblings:**
- `crevoke4b-bare-2026-11-25` is 4(b)'s kit (on silicon on 88db40027). It runs on this build too.
- `b5b2n-bare-2026-11-25`'s wrapper is used here byte-identical (`inc/`, as in crevoke4b).
- `sublet-life` is the control, and its image is byte-identical to crevoke4b's.

**Images:**
- `sublet-life`: the control, first.
- `b5b2-untagged`: CINCOFFSET, CINCOFFSETIMM and SCC on an untagged x[rs1] in protected U and in S, the result's first use, a tagged rs2, and plain M's CDERIVE_S inexact child.

The B5b-2 kit has never exercised this rule in U before. The image mints through CSMINT, and no image holds a custom-3 encoding (checked by build.sh).

`build.sh` builds with riscv64-unknown-elf-gcc; the recorder register comes from pick-regs.py (x8). `images/SHA256SUMS` pins the images. `PREREG.md` holds both trees' vectors and the grading. `expected-vectors.json` carries the same for graders. Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py`.
