# r61-stale-bare-2026-10-24

R-61 as bare JTAG arms for the batch bitstream (`caplifive_vp2c_c721c1a2a.bit`, sha256 8f4a486f2c96…85ed6e). R-61: inside a
virtual context, one REFUSED consuming LDC leaves the load unit's store-permission flag at 0, and a later load-unit
exception is relabelled STORE_PAGE_FAULT, including the silent R-43 replay of a LEGAL first-touch load, which then faults
with cause 15 (`docs/ref/ISSUES.md`, R-61). Siblings: Q-19 (the ruling the refusal implements), R-43 (the replay it
relabels), S-24 (a different cause-15 loop, fixed by the PTE's D bit).

`build.sh` rebuilds the three images from `tests/vb-trans.S` (capstone-ariane vp3-ringcut dcc6daf6e, checked against
`VP_SRC` when given) with the wrapper of `va-alloc-bare-2026-10-23` (inc/, also vp2-bare's); `images/SHA256SUMS` pins
them; `PREREG.md` holds the pre-registered vectors, simulated from THESE images on c721c1a2a's RTL and on the fixed
tree, and the grading rule. Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG` = the image, `SUP_REC_ADDR`
from its `.rec`), the bare-boot path only; no module, firmware or kit maker is involved.
