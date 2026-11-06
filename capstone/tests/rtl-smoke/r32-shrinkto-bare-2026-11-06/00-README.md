# r32-shrinkto-bare-2026-11-06

R-32's SHRINKTO half on silicon: how many bytes `SHRINKTO imm` gives the child capability. The RTL before capstone-ariane
a3b7a22c0 sets `rd_end = cursor + imm - 1` while its `end` is exclusive, which leaves imm - 1 bytes. QEMU
(`helper_csshrinkto`) and the fixed RTL leave imm bytes. The source line is present in C.5a (1fc2cd5aa), stage A
(c9c7952bd), bitstream 3 (e7af48457) and every earlier tree. Before this kit, the 63 bytes for SHRINKTO 64 had been
read in simulation only (capstone-ariane 95930b8e3). The fix rides on capstone-ariane stage-b-batch fe4810dd1 as
0ab365c26.

Why a readback: compiled code gets SHRINKTO with the compiler's +1 on the immediate (`-capstone-shrinkto-end-adjust=1`)
until the fix is in a bitstream. Under it, a pre-fix core gives N bytes and a fixed one gives N + 1, and both pass.
Only reading the bounds back tells the two apart.

One bare JTAG image, no new bitstream. `build.sh` rebuilds it from `tests/shrinkto-readback.S`, which was written for
this kit. The wrapper is `fp-rm-bare-2026-10-26`'s (inc/ copied unchanged). The recorder uses x18-x20, which the test
never names. `images/SHA256SUMS` pins the image. `PREREG.md` holds the vectors simulated from THIS image on the C.5a,
stage A and stage-B-batch trees. `r32-grade.py <uart.txt> [--expect prefix|fixed]` grades one boot from its capture.

Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG`, and `SUP_REC_ADDR` from the `.rec`), the bare-boot path
only.

Siblings: R-32 in `docs/ref/ISSUES.md`. The bound-convention entry is the decision-B record,
`docs/history/10-09-2026_19-00-00_decisions-A-and-B.md`.
