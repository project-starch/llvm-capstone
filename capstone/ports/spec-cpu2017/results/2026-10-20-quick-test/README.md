# Quick test, 2026-10-20: four short benchmarks, three arms, test inputs, one repetition

The first comparison on toolchain v4 (manifest `toolchain`, dev of 2026-10-20 plus 23 PRs): 502.gcc_r,
538.imagick_r, 544.nab_r and 519.lbm_r at the test size, every arm, one repetition, each run in its own
guest pinned to a socket-1 core on the build host, host load below the plan's bound throughout.
`report.md` is tools/compare-report.py's table; `records.txt` holds the plan, the CPU assignment, the
build record's SDK and compiler lines and the SDK recipe's proof lines.

Reading: the 7 s and 3 s runs (gcc, imagick test) are inside their noise; nab and lbm carry weight.
All ratios are guest time under QEMU; the per-access lifetime check runs in both arms and is not in them.
A pipeline test of the same four on the previous toolchain (q4's, with the byte-loop memset) read lbm
1.157 temporal and 1.143 scrub-only against spatial, the reason the comparison runs on the rebased tree.
