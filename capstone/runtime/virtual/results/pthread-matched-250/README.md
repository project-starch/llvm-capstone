# Matched pthread comparison for the mapping registry

50 runs each of `run-pthreads.py` against two builds, interleaved (the arm order alternates every
round), on one host, one kit, one QEMU, one firmware, kernel `Image` and rootfs. Only the module
and launcher differ.

- `sdev`: dev `4985df88e712` (the control).
- `sreg`: this change (tree equal to commit `4d96d938fd26`, pushed before the rebase onto a newer dev).
- Platform: QEMU `b290f5656600`, `svadu=false`, `x-capstone-supervisor-slots=32`, flat-table module.

Result: 0 failures in 50 for both arms. The earlier crash (64_joined_lifetimes, a revoked stack
capability) did not reproduce on the control either, so this rules out a high crash rate for this
change and says nothing about the earlier crashes: at a true rate of 20% the chance of seeing 0 in
50 is about 0.00001, at 10% about 0.005, at 5% about 0.08.

Files: `results.txt` (one line per run: arm, run number, exit code, wall time, failed checks; the
start and end lines carry the load average), `inputs.sha256` (hashes of QEMU, launcher, module,
domain image, firmware, kernel and rootfs for each arm), `arms.txt` (arm definitions), `one.sh` and
`drive.sh` (the scripts that ran them). Pair this with the earlier observation of the crash on dev
(4 of 20 and 5 of 20 runs on 2026-10-19), which has no record in the repository.
