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

## Addendum 1 (8 Nov 23:1x, before its board call): E0, the stage-B bitstream, on board 3 after the NO_HZ pair

E0 = caplifive_stageb_a481031ca.bit (sha256 aecb81b2…40f0; C.5a + R-32 fix + R-68 NI fix + DYN handshake trim + LDC
rs1). Sealed and graded PASS by the rtl lane (land/rtl-evariants a3df86684667: congestion 4, 0 loops, WNS −11.007,
routed clean; LUT 169,275). The +317 FF are attributed to retiming and register splitting, with no new state.
1. The rtl lane's 32-image gate (land/rtl-evariants 7439d0ca7853 for the vectors, 3bc486095d65 for the driver
   perf/e0gate/e0-gate.sh 3): VALUE fields equal E0's sim; TIMING fields within the grader's rule. Exit 0 keeps the
   board on E0. Any other exit stops the chain, and C.5a is flashed back.
2. Only after a PASS, two SPEC stages (drivers perf/e0/gt-run-e0-q32.sh and gt-run-e0-thp.sh, kits bkit37-e0-q32 and
   bkit37-e0-thp = bkit32-q32 and bkit29-thp with only the bitstream changed):

| # | Stage | Reference (board 3, C.5a, same kit and exe) | Prediction |
|---|---|---|---|
| E0a | gcc train01 dcnofp (exe 736f881f4958ac3b) | 3,924.9 s; in-context instret 12,869,971,613 | rc 0, specdiff 1/1, outputs equal; instret within ±0.01 % (the RTL does not change instruction counts); run 3,780-3,920 s |
| E0b | mcf test THP (exe 3633be7e2f887047) | 11,230.8 s | rc 0, specdiff, outputs equal; run 10,600-11,200 s (mcf's ldc+stc are ~169 per 1k instructions; the trim saves ~0.5 cycles per LDC on a hit in sim) |

An instret difference beyond ±0.01 %, or any output difference, is a finding about E0, not a speed result.

## Addendum 2 (8 Nov 23:3x): E0 waits for the lead's confirmation of the reflash; board 3 runs NO_HZ perlbench meanwhile

The E0 chain was not launched: flashing a new bitstream needs the lead's confirmation in this conversation, and it
is asked for. So that board 3 is not idle, its NO_HZ chain continues with the optional stage of the exit agent's
pre-registration (perf/exit-kernel PREREG-nohz.md, 93ecfc29648d, "Optional second board"): perlbench test c3g on
bkit35-nohz (stage perf/exitk/stages/perl-c3g-tk, C.5a, no new bitstream), against out-best-b2 (test command
33,024 s; `kcyc − rcyc` 1,670 s): test command −1,000 to −1,300 s, `kcyc − rcyc` ≤ 600 s, specdiff 2/2. Board 3 is
not board 2, so the comparison also crosses boards. If E0 is confirmed before this stage starts, the stage is
withdrawn and E0 runs instead.
