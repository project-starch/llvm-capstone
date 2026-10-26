# fp-divsqrt-bare-2026-10-26

Are `fdiv.d` and `fsqrt.d` correctly rounded on silicon? In simulation of bitstream 2's RTL (capstone-ariane `dcc6daf6e`'s
core, fpnew 0.7.0 with the MERGED divsqrt `fpu_div_sqrt_mvp`) they are not. Round-to-nearest-even takes its sticky bit
only from four extra quotient bits, never from the remainder (`core/cvfpu/src/fpu_div_sqrt_mvp/hdl/norm_div_sqrt_mvp.sv`:
`Mant_forround_D` at :342 / :354, `Mant_sticky_bit_D` at :385-387). So a true result a little above half an ulp is
rounded as a tie, to the even neighbour below it. Over 512 random cases this is 23 `fdiv.d` and 19 `fsqrt.d` results,
each 1 ulp toward zero. A bit model of that rounding predicts exactly those indices (an adversarial audit, 2026-10-26).
This kit asks whether the board reads the same bits.

Why it matters: the BEEBS programs `ludcmp` and `st` read VERIFY-FAILED on bitstream 2 and pass on the QEMU twin, whose
softfloat is correctly rounded. `st` checks exact equality against a correctly rounded sqrt
(`docs/design/capstone-softfloat-libm.md`).

`build.sh` rebuilds the images from `tests/fp-ieee-divsqrt.S`, a copy of capstone-ariane
`verif/tests/custom/capstone/fp-ieee-divsqrt.S` on vp3-ringcut, tests only above `dcc6daf6e`. Its table comes from
`tests/gen-fp-ieee-divsqrt.py` (seed 20261026); the `-DONECASE` block was added by hand afterwards and is the only
difference. The wrapper is `r58-r59-q24-bare-2026-10-26`'s (inc/ copied). `images/SHA256SUMS` pins the images.
`PREREG.md` holds the vectors simulated from THESE images on dcc6daf6e's RTL.

For the board: **`fp-all` only**, one cold bare boot. `fp-c3` and `fp-c15` run one case each behind 32 nops and a fence.
They are the simulation's evidence that each error is per operand, not per sequence, and are not needed on the board,
because `fp-all` reports the same two first misses.

Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG`, `SUP_REC_ADDR` from the `.rec`), the bare-boot path only.
