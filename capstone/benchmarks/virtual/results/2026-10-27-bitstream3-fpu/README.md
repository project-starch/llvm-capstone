# ludcmp, st and minver on bitstream 3 (R-62/R-63 fixed), T1 against T2

Bitstream caplifive_vp2c_e7af48457 (capstone-ariane e7af48457), board 2, one boot, 2026-10-27, 03:46–04:01.
- Kit `perf/main4/b3k1`, run-b3k1.sh sha256 cbf60f75e5eb79ae.
- The images, module (9c60fb53f80ce67e) and launcher are the bitstream-2 main run's.
- Pre-registration: PREREG-t1-t2-sampling.md, section "Bitstream 3: rerun of ludcmp, st and minver", pushed on
  runtime/bench-counters at 8bfc279da208 before any bitstream-3 boot.

**Identification** (`identification.txt`). The first step runs ludcmp and st natively, outside any context. These are
the same binaries that, on bitstream 2, read VERIFY-FAILED on 4 of 4 iterations with BENCH-DONE failed=4 and rc 1.
On bitstream 3 both read verify_failed=0, failed=0 and rc 0. So the board ran the fixed divider.

**Correctness.** All 24 processes pass in both arms.
- minver's check= is 0xd6cc5fbda167dc7b on every iteration, warm-ups included. That equals the QEMU twin; bitstream 2
  read 0x74ceeb9e3de2c383.
- ludcmp and st carry no hash. Their own verify passes.

**Timing.** In `per-program.tsv`, T2/T1 is ludcmp 1.0004 [0.9986, 1.0021], minver 1.0000 [0.9990, 1.0010] and st 1.0000
[0.9982, 1.0019]. None allocates in its timed region. These rows belong to bitstream 3 and are not pooled with
bitstream 2's suites.

**Regeneration**, from this directory:

    zcat runs-b3k1.tsv.gz > main.tsv && python3 ../../perf-analyze.py report main.tsv --out . --boot 10000 --seed 20261026 && rm main.tsv
