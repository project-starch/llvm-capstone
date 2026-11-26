# LSRH on silicon: an in-session LSHOFF, LSRH, LSHOFF2 bracket on b42c8c644 (pre-registration, 2026-11-26)

The eight-stage twin gate held (tests/spec8-lsrh, results a849e646a469): every stage of LSRH is within +0.05 % + r of
LSHOFF on the trap twin. Under the semantics owner's ruling, the board form returns only under a pre-registration of
its own, with its own imagick band and its control taken inside the board session. This is that pre-registration.
Written before any board image of this run was built or booted.

## Arms

All three arms are built by the lane's compiler (llvm-sinkland clang, sha256 b95bc7ce930075de), as in the twin run.
Under spec8's Amendment 1, B5TL1 is not the control: its compiler (llvm-c9) builds a different cpugcc_r, with one
swap of two instructions in mcf.o. That swap is in all three arms here. The session's B5TL1 rows are used only as
prices, never in the comparison.

| arm | natives | boot order |
|---|---|---|
| LSHOFF | perf/cclever/arms/LSHOFF, the twin control | first |
| LSRH | perf/cclever/arms/LSRH (-lsr-capability-exit-runtime-end -lsr-capability-exit-end-hoist) | second |
| LSHOFF2 | LSHOFF's natives, byte-identical images | third: the in-session repeat |

Native stripped sha256, first 16 digits:

| program | LSHOFF and LSHOFF2 | LSRH |
|---|---|---|
| perlbench_r | f3c8c80d71844564 | 9c6012e6538b6345 |
| cpugcc_r | 9908933454d63036 | 6b17ed3b7b30235a |
| x264_r | f686fe9679b04d69 | a91cd669417be917 |
| imagick_r | a666e331f10d73cc | 873462e6c4e97883 |
| nab_r | da57e5473a33e6fd | bb936adef2b57596 |

## Stages, and what is not booted

- **On:** gcc, imagick, perlbench (makerand only, as in g4mk), nab and x264.
- **lbm: off.** The twin decides it: LSRH's lbm image is LSHOFF's, byte for byte.
- **mcf and xz: off by default.** LSRH's images are LSHON's, and they read +0.001 % and +0.000 % on the twin. On
  b42c8c644 their times are UNKNOWN: on earlier bitstreams both ran past 3,600 s, with bounds of 30,000 s and 7,200 s.
  They come in only by an amendment that lands before the first boot of this run. That amendment needs two things:
  the board lane's g1b reading must price them on b42c8c644, and the session window must cover that price for all
  three arms.

## Platform

The campaign platform's Full row (plan 2026-11-25-line-b-final-campaign, the LSRH row):
- bitstream b42c8c644;
- keep-tags firmware with FW_FAST_RDTIME;
- the final Full kernel, NO_HZ_IDLE at HZ=100 (boot 13, runtime/b5b2-linux 098479a070eb).

**This is not the bring-up kernel the SPEC test rows ran on.**

## Images

The images use the campaign relink recipe (plan step 5): runtime/b5b2-linux 669e94f6eb71,
capstone/tests/rtl-smoke/b5b2-linux-2026-11-25/build/.
- That recipe sets the Full tree and FW_EXTRA="FW_FAST_RDTIME=y" (make-payload-final.sh), with the call pattern in
  build-campaign.sh.
- The campaign images themselves came from the same tree's payload script, called directly, and so will these.
- The stages are made by ports/spec-cpu2017/board/make-stage.py.
- Each arm's overlay is made from the Full campaign's qualification overlays (SP-B5TL1-<g>/ovl) by c4/spec8/board/make-board.sh:
  - every stage directory is byte-identical to its source apart from bin/native/*, and each native is the arm's
    strip.dom;
  - the script checks both, and its check was negative-tested: one edited run.sh and one swapped native each gave a
    MISMATCH and a nonzero exit.
- **The S image joins four campaign groups (g3, g2i, g4mk, g1b) under one init:** g2i's, which reads /stages.order and
  /stages.bound. g3 and g4mk used an earlier init without either file. All three arms share this init, so it is not
  a difference between them.
- The compiler lane builds the images.

**A kernel relink is not byte-reproducible, so an image is checked by what it carries, not by its hash:**
- its overlay manifest against the Full campaign image's, with bin/native/* the only difference;
- the kernel's .config keys;
- the fast-path count, which must be 2.

**The image and overlay hashes, with those three checks, go into Amendment 1, which lands before the first boot.** No
image exists as this is written.

The campaign's own Full arm is B5TL1, built by llvm-c9. It is a different build from LSHOFF here, and no reading of
this run is compared against it. The compiler identity (b95bc7ce930075de) is printed beside every number.

| image | stages, in this order | per-stage bound |
|---|---|---|
| S-LSHOFF, S-LSRH, S-LSHOFF2 | gcc, imagick (run, then val), perlbench, nab | 3,600 s each |
| X-LSHOFF, X-LSRH, X-LSHOFF2 | x264 (gen, then run) | 43,200 s (as spectest Amendment 11) |

The stage order is the same in all three arms, and every image boots fresh. Boot history is a variable here. On
28fcbb646, imagick ran in 20.6 s after x264 in the same boot; the first imagick on b42c8c644 took 11.6 s. Those two
are different bitstreams, so this is not a measured effect, but it is why no arm shares a boot with another.

## The session

- **One board, one bitstream, no reflash between the three boots of a bracket:**
  - the S bracket is S-LSHOFF, S-LSRH, S-LSHOFF2, back to back;
  - the X bracket is X-LSHOFF, X-LSRH, X-LSHOFF2, one per boot, in the board's longest unattended windows.
- **A reflash, or another bitstream's boot inside a bracket, voids that bracket, and it restarts.** Other images on
  the same bitstream may run between brackets, but not inside one.

## Prices

Every price below was read on the bring-up kernel with FW_FAST_RDTIME off (c4/spec8/board-b5tl1-times.txt).
Campaign-kernel times are unread.

| stage | price per arm | source |
|---|---|---|
| gcc | 5.8 s | b42c8c644, B5TL1-g3 |
| imagick | 11.6 s run + 5.6 s val | b42c8c644, B5TL1-g2b |
| perlbench (makerand) | UNKNOWN on b42c8c644 | g4mk pending |
| nab | UNKNOWN on b42c8c644 | g1b pending |
| x264 | gen 2,127 to 2,217 s, plus a run of more than 12,180 s | 28fcbb646 and b42c8c644; spectest Amendment 11 |

- **x264 needs at least about 14,300 s per arm, so at least about 43,000 s (12 h) for the X bracket.** Its upper
  bound is unread, since every b42c8c644 run so far was cut.
- The S bracket is three boots of a few minutes each, plus perlbench's and nab's unknown times.

## The reading

A stage's time is up1 - up0 on its run line (@@SPEC-END ... run ...), from /proc/uptime at 0.01 s.
- The stage prints no cycle or instruction counter on the trap kernel (sys= reads 0).
- The helper phases are not compiled by this compiler and are not part of any stage's reading: x264's gen (ldecod)
  and imagick's val (imagevalidate_538) are plain RISC-V binaries.
- The twin's x264 stage replaced ldecod with a copy of the host-decoded BuckBunny.yuv, so its count covers x264_r
  only. That matches the board's run phase.
- The twin's perlbench stage also ran test.pl, through a copy of the native named .perl-native. The count matches
  processes by the name perlbench_r_base, so it covers the makerand command only. That is the one command the board's
  g4mk stage runs.

## The binding gate, one-sided, per stage

For stage s, with t the run-phase time in each arm and spread_s = |t(LSHOFF2) - t(LSHOFF)|:

    t(LSRH) - t(LSHOFF) <= 0.0005 * t(LSHOFF) + spread_s + 0.02 s

**0.02 s is the print quantum, two readings of 0.01 s each.** The runtime lane accepted the term before any boot. It is
the instrument's resolution, symmetric, and favours no arm.
- Without it, the gate on imagick (0.05 % = 5.8 ms) and gcc (2.9 ms) is below the instrument's resolution, and a
  single tick of rounding would decide it.
- **What it means:** on the board, imagick and gcc are resolved only to about ±0.17 % (0.02 s of 11.6 s) and ±0.35 %
  (0.02 s of 5.8 s). There the board confirms the twin's instruction gate to within its print quantum and adds no
  finer claim.
- nab, perlbench and x264 are long enough that the term is small.
- The comparison is against the first LSHOFF reading, as written, not against the mean of LSHOFF and LSHOFF2.

**The failing reading:** any on-stage whose LSRH time exceeds the right-hand side.

**UNDECIDED, not a pass,** for a stage if any of these hold:
- in any of the three arms, the stage faults, reads NO READING, is cut at its bound, or fails specdiff;
- its output hashes differ between arms;
- the drift control below fires.

## Drift controls

The val and gen phases run identical binaries on identical inputs in all three arms, so they measure the board, not
the compiler.
- **S bracket:** imagick's val time in S-LSRH must lie within [min - q, max + q], where min and max are its
  S-LSHOFF and S-LSHOFF2 readings and q = 0.0005 * max + 0.02 s.
- **X bracket:** x264's gen time in X-LSRH must lie within the same window built from X-LSHOFF and X-LSHOFF2.

**If it lies outside, the boot drifted in a way the bracket does not capture.** Every stage of that bracket is then
UNDECIDED.

## What decides

- **All five on-stages hold:** LSRH enters the campaign as the plan's conditional compiler row, B5TL1 rebuilt with
  LSRH.
- **Any on-stage fails:** LSRH does not enter the campaign, and the failing stage and its reading are recorded.
- **A stage is UNDECIDED:** its bracket is repeated once, under the same rules. A second UNDECIDED leaves LSRH out of
  the campaign, recorded as undecided, not as failed.

## Predictions: CPI-converted points, prediction only

**ASSUMPTION: the instructions LSRH adds or removes run at the stage's average CPI.** So
Δt = Δinsn × t(LSHOFF) / N, with N the twin's LSHOFF count for the stage's native and Δinsn its LSRH - LSHOFF on the
twin (tests/spec8-lsrh RESULTS). This is not a measured CPI: no P1 CPI and no per-function CPI goes in. These points
do not bind.

| stage | twin Δinsn | Δ/N | predicted Δt |
|---|---:|---:|---|
| imagick | -1,676,067 | -1.774 % | -0.21 s of 11.6 s, the only stage where the prediction clears the quantum |
| x264 | +20,594,314 | +0.004 % | about +0.5 s at 12,180 s, against a 6.1 s band before the spread |
| nab | -2,984,712 | -0.042 % | UNKNOWN, since nab's time is unread; -0.042 % of it |
| gcc | +696 | +0.003 % | 0.0002 s, below the quantum |
| perlbench | -34 | 0.000 % | 0, below the quantum |

- For gcc and perlbench the prediction is "no regression beyond the spread", not a value.
- imagick is the bracket's positive control: a reading of t(LSRH) - t(LSHOFF) between -0.30 and -0.12 s is
  predicted. **If imagick reads no reduction at all, the instruction-count reduction does not reach time on this
  board.** That would not fail the one-sided gate, but it would be recorded as a refuted prediction.

## What this does not decide

- P1, where the run-time endpoint does not hold.
- Train inputs.
- Any comparison against B5TL1's rows.
- mcf and xz, unless Amendment 1 adds them.
