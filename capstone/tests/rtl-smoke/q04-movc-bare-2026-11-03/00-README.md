# q04-movc-bare-2026-11-03

This kit checks Q-04 (b) on silicon: MOVC keeps an untagged source. It is one bare JTAG arm for bitstream 3
(`caplifive_vp2c_e7af48457`) and the ObjBounds variant (`caplifive_vp2c_603528b8b`).

- The image is capstone-ariane `movc-untagged-source.S` (q04-board b534e60ca), built with `-DBOARD_INTS` so that both
  MOVC semantics return a whole vector.
- `PREREG.md` holds:
  - the source facts;
  - the two vectors, both simulated from this image: PRESENT on both sealed trees, and ABSENT on c7e6f604f, the field
    RTL;
  - the grading rule.
- `grade/q04-grade.py` applies the rule.
- `build.sh` rebuilds the image with `objb-bare-2026-10-30`'s wrapper (`inc/` copied from it), and
  `images/SHA256SUMS` pins it.
- The runner is `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG` = the image, `SUP_REC_ADDR` = 0x80001140).

Siblings:
- `hpm-bare-2026-11-03`, which runs in the same boot gap; its hpm-5 is the boot control;
- the C-32 lead decision, which this reading informs.
