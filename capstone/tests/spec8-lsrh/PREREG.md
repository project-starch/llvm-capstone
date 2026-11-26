# LSRH on B5TL1's eight SPEC test stages, with an LSHOFF repeat on the same init (pre-registration, 2026-11-26)

The spec8 split (tests/spec8-split, results 1430c473c2c6) found two things:
- -capstone-pool-base-at-entry carries spec8's two regressions;
- LSRH (-lsr-capability-exit-runtime-end -lsr-capability-exit-end-hoist) costs neither nab nor perlbench and keeps
  imagick's reduction.

The semantics owner's ruling: a board form returns only for a sub-option that holds +0.05 % on all eight twin stages,
pre-registered with its own imagick band and a control band from an LSHOFF repeat on the eight-stage init. This is
that twin run for LSRH. The board form, if this holds, gets its own pre-registration afterwards.

Written before either boot.

## Arms and images

The compiler is llvm-sinkland clang, sha256 b95bc7ce930075de, the binary spec8 and the split used. The arms are
built by build-arm.sh.

| arm | images | boot |
|---|---|---|
| LSHOFF | spec8's, already booted (spec8 8abd11010c5b) | not re-run |
| LSRH | perf/cclever/arms/LSRH | new |
| LSHOFFR | LSHOFF's images, byte-identical (images.sha equal) | new: the repeat |

LSRH stripped sha256, first 16 digits:

| program | hash | identity |
|---|---|---|
| perlbench_r | 9c6012e6538b6345 | |
| cpugcc_r | 6b17ed3b7b30235a | |
| mcf_r | 92bd6c83f94070e9 | = LSHON |
| lbm_r | 0d9105dab2bccdfe | = LSHOFF |
| x264_r | a91cd669417be917 | |
| imagick_r | 873462e6c4e97883 | = spec8's lsrh image |
| nab_r | bb936adef2b57596 | |
| xz_r | 69fdbd33b2ba851d | = LSHON |

## Procedure

spec8's procedure and platform, unchanged:
- qemu-crv, fw-y, kernel 749fed2606e3ebca, -m 963M, capmix3 at shift 4;
- the same init and stages, built by spec8's make.sh, which now takes an optional image arm;
- the same reader, count8.py, which now compares the first arm against each other arm. Its output on spec8's two
  arms is unchanged.

There are two boots, each taking and releasing the QEMU lock: LSRH, then LSHOFFR. Each stage is killed 10,800 s
after it starts, and a stage cut in either boot is void.

## The control band

For each stage, the repeat difference is r = |LSHOFFR - LSHOFF|. The gate for LSRH is LSRH - LSHOFF ≤ +0.05 % of
LSHOFF + r. That is the +0.05 % band widened by the cross-boot spread this init actually shows.

## Predictions

1. **Repeat: LSHOFFR equals LSHOFF within ±1,000 on seven stages.** The split read cross-boot spreads of 0 to 90 on
   nab, imagick and perlbench, and spec8 read lbm 0 and xz +24 across two different images. **x264 has never been
   repeated.** Its band is written, not derived: within ±0.001 % (±5.07 M).
2. **Identity rows.**
   - lbm: LSRH - LSHOFF is within ±1,000, because the image is LSHOFF's.
   - mcf and xz: LSRH equals spec8's LSHON reading within ±1,000, because the images are LSHON's (mcf
     26,644,603,084, xz 49,619,823,292).
3. **nab -2,984,712 ± 1,000 and perlbench -34 ± 1,000.** These are the split's readings of the same images. Both
   counts were the same under spec8's init and the split's.
4. **imagick: -1,676,157 ± 1,000**, spec8's reading of this image under this init. The board's imagick band is
   written from this number.
5. **x264: between 0 and +0.010 % (0 to +50.7 M).**
   - LSRH changes 10 x264 functions. One is hot: x264_me_search_ref, about 5.48 G instructions of 506.6 G, which is
     LSRH-only and 25 instructions longer.
   - spec8's LSHON read x264 +20,595,573 (+0.004 %), and +18.6 M of it in that function at shift 4.
   - The other nine changed functions run under 4 × 10^4 instructions in LSHOFF.
6. **gcc: within ±0.05 % (±10,855), no direction.** LSRH changes 178 functions, which run about 1.35 M instructions in
   LSHOFF at shift 4. spec8's LSHON read +707.
7. **Outputs: every stage's hash set in LSRH and LSHOFFR equals LSHOFF's.**
8. **The gate: every stage of LSRH is within +0.05 % + r of LSHOFF.** By predictions 2 to 6, all eight hold, and
   the largest is x264 at about +0.004 %.

## What this does not decide

- The board form, which gets its own pre-registration after this.
- Cycles. These are instruction counts on the trap twin.
- P1, where the run-time endpoint does not hold.
