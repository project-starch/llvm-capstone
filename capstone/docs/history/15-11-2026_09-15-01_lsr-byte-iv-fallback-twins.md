# LSR byte-IV fallback (llvm-c13): twin results on the eight SPEC rows, the directed test, and the x264 validation gap

Results for `-lsr-capability-byte-iv-fallback`, commit 2c28b079739d on `capstone-loop-offset-gep`. That commit
gives the gates. This note gives the runs it defers: the SPEC twins and the directed test `loop-offset-gep`. The
build is llvm-c13 = 29027126f39b plus that commit. Its SDK arms:
- `perf/cclever/arms/FRCDKMSa3hy2`: FRCDKMSa3 plus the option;
- `FRCDKMSa3o2`: the same compiler without the option;
- the reference `FRCDKMSa3`: llvm-c9.

## Builds

- **Off.** `FRCDKMSa3o2` is objcmp-identical to FRCDKMSa3 (rc=0), and all 11 stripped images are byte-identical.
- **Two-pass property** (`perf/lgep/groups-equal.py`, rc=0, 0 breaks on every row). For every loop, the
  capability-group count with the option equals the count without it. The byte-form groups the option adds:

| row | loops off / on | cap groups (both) | byte groups added |
|---|---:|---:|---:|
| 500.perlbench_r | 190 / 308 | 209 | 127 |
| 502.gcc_r | 1939 / 2497 | 2333 | 738 |
| 505.mcf_r | 6 / 15 | 7 | 13 |
| 525.x264_r | 861 / 1000 | 1295 | 242 |
| 538.imagick_r | 380 / 1507 | 474 | 1214 |
| 544.nab_r | 107 / 144 | 171 | 49 |
| 519.lbm_r | 8 / 11 | 10 | 3 |
| 557.xz_r | 89 / 119 | 119 | 34 |

- **Against c12's one-pass build** (FRCDKMSa3hy):
  - byte-identical: lbm (e17b45f589462b6d), xz (6c1c8d9c88eeeb62) and perl (8cf050818865bc80);
  - different: mcf, gcc, x264 and ldecod, imagick, nab.
  The identical rows keep c12's twin. The others were twinned again on c13.

## Directed test (`chain-dt2.sh`, native twin, slot 29, images-a0ed with the board's domain flags)

For both c12 and c13, off against on, the six loop shapes (fwd, bwd, shifted, negidx, field, idx32) return equal
values and exit 0. The in-window control exits 0. The out-of-window control (`ctl-out`) dies with signal 11 and
cause 24 in both arms, as designed. `judge-lgep.py` PASS for both. The first attempt, inside `chain-c13.sh`, is void:
it ran before that chain set `TWIN_CMDLINE_MAX=1024`, which the domain flags need, left no `results/inp.out`, and
judged rc=1. DT2 replaces it.

## SPEC twins

QEMU in-context user instructions at shift 1, from `perf/lgep/twin-table.py`. Every row is measured against the
camp base BIb (llvm-c9 FRCDKMSa3). The table counts the windows of each run, and missing data is an ERROR, never a
zero.

| row | c13 user vs base | outputs |
|---|---:|---|
| perl full (st-perl-hy2 vs s1-FRCDKMSa3, board-faithful stage) | -483,880,297 (-0.553 %) | identical, specdiff 2/2 |
| perl test-sw (c12's twin, image identical) | -704,991 (-0.018 %) | identical, specdiff 1/2 as base |
| gcc train01 | -844,006 (-0.007 %) | identical 1/1 |
| mcf | -71,697,786 (-0.269 %) | identical 2/2 |
| x264 | +149,960,337 (+0.028 %) | identical, host imagevalidate 3/3 |
| imagick | +264,520 (+0.171 %) | identical 1/1 |
| nab | -21,368 (-0.000 %) | identical 1/1 |
| lbm (c12's twin, image identical) | +1,706,778 (+0.028 %) | identical 1/1 |
| xz (c12's twin, image identical) | +10,918,894 (+0.022 %) | identical 12/12 |

- **Kernel.** Every row's kernel delta is within ±0.35 M, except perl full: +2.09 M against -484 M user.
- **Global `-lsr-capability-byte-iv`** (arm BIo, llvm-c9) gives every group the byte form, and its twins are
  worse: x264 +6.47 %, lbm +6.43 %, xz +0.48 %, gcc +0.41 %. The fallback keeps the default where the default
  reduces, which avoids those losses.
- **Silicon** is the SPEC lane's to run (spec-silicon 08891d244c0d, Addendum 130):
  - perl full on board 3, after FRCDKMSa3's full perl on the same board;
  - mcf as a matched pair on board 2.
  - Adoption: perl at least 0.2 % faster, and mcf not slower than +0.15 %.

## The x264 twin never validated: a stage gap, not an output defect

On every arm, base included, x264's twin validation read `specdiff 0/3`, and its output column read DIFFER. Both
come from the instrument:
- **Validation.** The twin stage `perf/nat-fair/stages/525.x264_r-test` was built with make-stage.py's
  `--no-target-validation`, the board-image option. Its plan has `target_validation: []` and `refs: {}`, and
  run.sh prints `validations=0`. So imagevalidate never runs, `imagevalidate_frame_*.out` is never written, and
  the host specdiffs open missing files.
- **DIFFER.** The only result file that differs between arms is ldecod's `log.dec`, and only in its Date and Time
  fields. Every encoder output is byte-identical on all four arms (BIb, BIo, HYo, HY2o): BuckBunny_New.264
  0e453c440e68 and frames 0, 50, 100, 150 and 155.

The host check that replaces it:
- SPEC's `ImageValidator.c`, built with host gcc into `perf/lgep/iv/`, runs with runcpu's compare.cmd arguments
  (`-avg -threshold 0.5 -maxthreshold 20` against `data/test/compare/frame_N.org.tga`), then
  `specdiff -m -l 10 --reltol 0.085`.
- All three frames pass (SSIM 1.000000000).
- Control: frame_100 checked against frame_50's reference fails (SSIM 0.326, 10232 blocks below the threshold,
  specdiff rc=1).
- specdiff compares the file name printed in the line, so the frame must be passed under its bare name.

`twin-table.py` now does this for every x264 row, with the control run on each call as a must-fail. It also masks
only log.dec's Date and Time fields, asserting one mask per data row. Negative tests, all fired:
- a time-only difference hashes equal;
- a changed SNR field hashes different;
- a broken row format is an ERROR;
- a substituted frame reads FAIL;
- a control that passes is an ERROR.

With these, x264 reads "identical, host-iv 3/3" on all four arms. Rebuilding the twin's x264 stage with target
validation is in the SPEC lane's backlog; the board image keeps the option for size.
