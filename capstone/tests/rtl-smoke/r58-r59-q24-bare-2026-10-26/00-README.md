# r58-r59-q24-bare-2026-10-26

Silicon acceptance for three fixes that are in RTL on every bitstream since the batch `c721c1a2a` and were accepted in
simulation only: **R-58** (the CSRUNV caller gets its own srevroot back after a context escapes), **R-59** (the CSRUNV
precheck refuses a lifetime-table word with any of bits [11:0] set) and **Q-24** (firmware may delegate a capability cause
to S through medeleg). Bare JTAG arms for bitstream 2 (`caplifive_vp2c_dcc6daf6e.bit`), no new bitstream.

`build.sh` rebuilds the six images from `tests/` (va-runv.S and va-srevroot.S at capstone-ariane dcc6daf6e, q24-deleg.S at
vp3-ringcut, tests only above dcc6daf6e; checked against `VP_SRC` when given) with `r61-stale-bare-2026-10-24`'s wrapper
(inc/ copied); `images/SHA256SUMS` pins them; `PREREG.md` holds the vectors simulated from THESE images on dcc6daf6e's RTL.
Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG`, `SUP_REC_ADDR` from the `.rec`), the bare-boot path only.
Siblings: R-58, R-59 and Q-24 in `docs/ref/ISSUES.md`.
