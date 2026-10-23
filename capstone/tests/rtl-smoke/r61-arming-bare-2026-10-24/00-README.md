# r61-arming-bare-2026-10-24

R-61's two arming conditions as bare JTAG arms for the batch bitstream (`caplifive_vp2c_c721c1a2a.bit`). R-61 itself, a
REFUSED consuming LDC followed by a legal first-touch load faulting with STORE_PAGE_FAULT, is confirmed on silicon by
`r61-stale-bare-2026-10-24` (`docs/ref/ISSUES.md`, R-61). This kit asks whether a refusal is needed at all: the load
unit captures the page's store permission (`store_ok = w && d`) on EVERY LDC, consuming or not, so in simulation a
NON-consuming LDC from a read-only page (stale3), or from a WRITABLE page whose D bit is still clear (stale4), arms the
same stale flag. stale0 is the control: the same legal load with no LDC before it.

`build.sh` rebuilds the four images from `tests/vb-trans.S` (capstone-ariane vp3-ringcut 0311ca24a, tests only above the
RTL under test) with `r61-stale-bare-2026-10-24`'s wrapper (inc/ copied from it); `images/SHA256SUMS` pins them;
`PREREG.md` holds the vectors simulated from THESE images on c721c1a2a's RTL and on the fixed tree. Runner:
`sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG`, `SUP_REC_ADDR` from the `.rec`), the bare-boot path only.
