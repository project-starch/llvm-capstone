# Board plan from 8 Nov 22:1x: Plain train on one board, quick tests on two (pre-registration)

Decision taken with the external collaborator (8 Nov 22:0x): one FPGA measures Plain train for all of SPEC and that
number is locked. The other two FPGAs spend about a day comparing faster Capstone configurations on quick programs
(the module/kernel line now, native on its new bitstream when it is sealed). The Temporal (Capstone) train run then
starts on the faster configuration. It replaces the board split of results/2026-11-08-train for boards 2 and 3:
their queued Capstone train chains were cancelled at 22:1x, before either started.

**Timing, stated for the decision:** Plain train alone is ~77 board-hours (perlbench 13.5, nab 14.5, mcf 12.7,
imagick ~12, x264 11.3, lbm 7.8, xz 4.8 h; gcc train01 is already measured). On board 1 from 20:06 Sunday it ends
about Thursday 01:00 SG. A Capstone train on two boards is ~45 h from its start.

## Board 1: Plain train, every benchmark (results/2026-11-08-train's stages and gates T0-T2)

Running since 20:06 (perlbench). The list now holds perlbench, nab, mcf, imagick, x264, lbm and xz (Plain-with-cache,
spec/stages-train-fair). gcc is skipped by an existing output directory: it is already measured.

## Board 3: the tick-replay lever K1 (NO_HZ_IDLE board kernel), after its running perlbench test

The predictions are the exit agent's, pushed before any board use: branch perf/exit-kernel, PREREG-nohz.md
(93ecfc29648d) and its QEMU addendum (2d845189ceea), N0-N5 and P1. Kit t3-smoke/bkit35-nohz, driver
perf/exitk/board/gt-run-nohz.sh, stages gtr-c3f-tk (Capstone gcc train c3f) then plain-gtr-tk (Plain gcc train,
the control that says whether the kernel moves Plain). About 2.2 h.

## Board 2: kfast (C1), after its running Plain-reuse x264 test

Kit t3-smoke/bkit35-kfast (bkit30-unify with module 40e2e0a9fd831a53 and launcher 02f1a98eaf4d209d, kfast=1;
results backlog C1), driver perf/univmod/un-run-kf.sh. The module is new, so its first stage is the Tier 1 run.

| # | Stage | Prediction (reference: the same stage on bkit30-unify, board 2) |
|---|---|---|
| K0 | gcc train01 unify-rgz-nd (Tier 1) | rc 0, specdiff 1/1, outputs equal out-nd-b2's; control rungs pass; run within ±2 % of 4,518.65 s (gcc makes 156 delegated calls, so kfast serves almost none) |
| K1 | perlbench test unify-rg-nd | rc 0, specdiff 2/2, test.out 0 FAILED; kind-3 exits fall 25-33 % (QEMU: 29 %); run 29,100-30,100 s against 30,120.3 s (out-nd-b2) |

A K0 failure stops board 2's chain before K1.

## What follows (Tuesday)

The native bitstream (E5 + CSRUNV 4/5 detach/adopt, rtl lane) is due sealed Tuesday morning. Its quick tests, and the
choice of the configuration for the Temporal train, get their own pre-registration.
