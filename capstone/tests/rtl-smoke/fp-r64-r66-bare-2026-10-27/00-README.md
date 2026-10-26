# fp-r64-r66-bare-2026-10-27

Silicon readings for R-64, R-65 and R-66 (docs/ref/ISSUES.md), the divide/sqrt unit's flag, overflow,
subnormal-carry and NaN-boxing defects. The siblings R-62 and R-63 are fixed on bitstream 3; see
fp-divsqrt-bare-2026-10-26 and fp-rm-bare-2026-10-26. These three are present on bitstream 3 by source and fixed on
capstone-ariane r64-flags, which is not yet in a bitstream.

Five bare M-mode images (`images/*.bin`, load at 0x80000000; `images/*.rec` gives `board_rec`), one cold boot each.
`build.sh` rebuilds them. PREREG.md holds both vectors and the grading rules.
