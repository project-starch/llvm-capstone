# fp-rm-bare-2026-10-26

R-63 on silicon: under RDN, fdiv and fsqrt round toward +inf, under RUP toward -inf, and RMM truncates. fpnew hands its
rounding-mode encoding to the divsqrt unit unmapped (`fpnew_divsqrt_multi.sv:101`, `:285`); the unit numbers PLUSINF 2
and MINUSINF 3 (`defs_div_sqrt_mvp.sv:20-21`) and has no arm for RMM (4). One bare JTAG image for bitstream 2
(`caplifive_vp2c_dcc6daf6e.bit`), no new bitstream.

`build.sh` rebuilds the image from `tests/fp-rm-divsqrt.S`, a copy of capstone-ariane
`verif/tests/custom/capstone/fp-rm-divsqrt.S` at r62-divsqrt e7af48457 (tests only; checked against `VP_SRC` when
given). The wrapper is `fp-divsqrt-bare-2026-10-26`'s (inc/ copied). `images/SHA256SUMS` pins the image. `PREREG.md`
holds the vectors simulated from THIS image on bitstream 2's RTL and on the fixed RTL.

Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG`, `SUP_REC_ADDR` from the `.rec`), the bare-boot path only.
Siblings: R-62 and R-63 in `docs/ref/ISSUES.md`; the double-precision rounding kit is `fp-divsqrt-bare-2026-10-26`.
