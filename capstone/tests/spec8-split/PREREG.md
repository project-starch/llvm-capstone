# Which option costs nab and perlbench: pool-only against lsr+hoist-only at shift 1 (pre-registration, 2026-11-26)

spec8 (tests/spec8-twin-count, results 8abd11010c5b) measured nab_r +0.340 % and perlbench_r +0.369 % under LSHON,
which carries three options. The semantics owner ruled that LSHON does not go to the board, and approved this split.
It runs one option at a time, at the profiler's instruction resolution.

Written before any boot of this run.

## Arms

All arms are built by build-arm.sh from B5TL1's SPEC build records with B5TL1's flags. The compiler is
llvm-sinkland clang, sha256 b95bc7ce930075de, built 01:40, before every arm here. That is the binary spec8 used.

That binary's --version names cf639153be7b, not the lane head 3eac93f488ec. LLVM's revision header is not
regenerated in this worktree, so it is identified by its hash. It accepts -lsr-capability-exit-end-hoist and the
option changes code, so it carries the hoist.

| arm | extra options | nab_r | imagick_r | perlbench_r |
|---|---|---|---|---|
| LSHOFF | none | da57e5473a33e6fd | a666e331f10d73cc | f3c8c80d71844564 |
| POOL | -capstone-pool-base-at-entry | d9915445c6fc495e | f8dfaf92927c6db8 | 4b3d87cfde1b64f1 |
| LSRH | -lsr-capability-exit-runtime-end -lsr-capability-exit-end-hoist | bb936adef2b57596 | 873462e6c4e97883 | 9c6012e6538b6345 |
| LSHON | all three | 27a7133d719bef1f | 06a0668a4535e67b | d1b6373ff014d9ba |
| LSHOFF2 | LSHOFF's images, booted a second time | | | |

The hashes are stripped sha256, first 16 digits.
- LSHOFF and LSHON are spec8's images.
- Two rebuilt images equal earlier builds byte for byte: POOL imagick equals the loop-shape pool image, and LSRH
  imagick equals spec8's lsrh image.

### Image identity across all eight programs

| program | equal to LSHOFF | equal to LSHON |
|---|---|---|
| mcf_r, xz_r | POOL | LSRH |
| lbm_r | LSRH | POOL |
| others | none | none |

## Static reading, before the boot

These were read with c4/spec8s/fnsig.py, which compares each function's instruction sequence with immediates and
addresses removed, decoded with +m,+a,+f,+d,+c. A function that is unchanged in an arm executes the same instructions
on the same data.

**nab_r.** 20 functions change in LSHON. POOL changes 16, LSRH 5. Each changed function in LSHON equals one option's
version, except md and cap, which both options change.
- POOL alone changes mme34 (+16 instructions), ephi, eangl and rot4p.
- LSRH alone changes matrix, imatrix and putbnd.

**perlbench_r.** 51 functions change in LSHON. Every function makerand.pl executes more than 10^4 times that changes
is a POOL change, and LSHON's version equals POOL's: Perl_pp_modulo, Perl_pp_int, Perl_cast_uv and Perl_my_atof2.
The functions LSRH changes run about 2,500 instructions in spec8's LSHOFF profile (Perl_av_store, av_reify, pad_tidy,
av_clear). The hot interpreter ops (pp_nextstate, pp_padsv, pp_and, pp_const) are unchanged in every arm.

**The hypothesis: -capstone-pool-base-at-entry carries both regressions.** The run tests it.

## Procedure

Five boots under the QEMU lock, one per arm, in the order LSHOFF, POOL, LSRH, LSHON, LSHOFF2. The platform is
spec8's: qemu-crv, fw-y, kernel 749fed2606e3ebca (pinned copy), -m 963M, sublet.dom_binfmt=1. The only change is
capmix3 at shift 1, which gives one bucket per instruction.

Stages, from spec8's make.sh (c4/spec8s/make.sh):
- nab, imagick and perlbench, in that order, from B5TL1's board stages;
- perlbench without its test.pl command. spec8 did not read that command, which is not a bin/native command, and it
  took 1,697 of the boot's 8,169 s.

Reading: count8s.py, spec8's count8.py at shift 1, with each arm's delta read against LSHOFF.
- Per-function counts: each instruction's bucket is credited to its own function. A shift-4 split could not do this.
- Errors are as in spec8: a command with no SNAP pair, no enters line or a zero count is an error.

## Predictions

1. **Repeat: LSHOFF2 equals LSHOFF within ±100 per stage.** The images and the init are the same. This band is the
   control for any later pre-registration on this init.
2. **Reproduction: LSHON minus LSHOFF equals spec8's within ±10,000 for nab (+24,313,900) and perlbench
   (+11,407,013), and within ±1,000 for imagick (-2,775,976).** The init differs from spec8's, but the offset
   cancels within one init.
3. **Per-function attribution: in every arm, each function unchanged against LSHOFF counts the same as in LSHOFF.**
   The sum of those functions' deltas is within ±1,000 per stage. This checks the shift-1 reader: a function's code
   and data are the same, so its count must be.
4. **perlbench_r:**
   - LSRH minus LSHOFF is within ±10,000;
   - POOL minus LSHOFF equals LSHON minus LSHOFF within ±10,000;
   - the whole delta lies in POOL-changed functions.
5. **nab_r:**
   - POOL minus LSHOFF is at least +20 M;
   - LSRH minus LSHOFF is within ±5 M;
   - the POOL and LSRH deltas sum to LSHON's within ±1 M, the slack covering md and cap, which both options change;
   - mme34's delta in POOL equals its delta in LSHON exactly, because the two arms' mme34 is the same code.
6. **imagick_r, which serves as the positive control that the split shows each option's effect:**
   - POOL -1,099,810 ± 1,000, the loop-shape count of this same image;
   - LSRH -1,676,157 ± 1,000, spec8's lsrh reading of this same image.
7. **Outputs: every stage's hash set is equal across the five boots.**

## What this does not decide

The board form. Under the semantics owner's ruling, a board form needs its own pre-registration: a sub-option holding
+0.05 % on all eight twin stages, with an imagick band and a control band written before that boot. If prediction 4
holds and LSRH is within +0.05 % on nab, LSRH is that candidate. Its eight-stage boot, with an LSHOFF repeat on the
same init, will be pre-registered separately.
