# Compiler evaluation on silicon, step 1 (dev compiler) and step 2 (+movc-keeps-integer-source): pre-registration

This file is written and pushed before any of these boots.

**What is compared.** There are three image sets. Each comes from the SPEC test board tree's recipe (`rec-board`,
kit tools/build.sh without layout, all 8 benchmarks).

| Set | Tree | SDKs | Compiler |
|---|---|---|---|
| baseline | `spec/tree-board` (the 3 Nov board runs) | perf/sdk-T1, perf/sdk-T3 | cc-prmerged fca004e7e063 |
| dev | `spec/tree-boarddev` | c1-gate/sdk-dev2, sdk-dev-T3 | llvm-c1 = origin/dev 724bf0bea11e |
| devmk | `spec/tree-boarddevmk` | c1-gate/sdk-dev2-mk, sdk-dev-T3-mk | llvm-c1, with +movc-keeps-integer-source |

- **baseline.** Rebuilding it with this recipe (`tree-boardrepro`) reproduces all 22 of rec-board's executables byte
  for byte.
- **dev.** The compiler is the only change. sdk.json differs only in cc/ld and the SDKs' own library paths.
- **devmk.** The codegen flag is the only change from dev. The compiler lane's gate5 reads identical to dev on hosted,
  fp (2 slots), malloc and nodes.
- Step 1 is dev vs baseline. Step 2 is devmk vs dev.
- devmk may run only on a bitstream whose Q-04(b) reads PRESENT.
  - Bitstream 3 reads PRESENT (board 1, 3 Nov 19:26, recorder; rtl lane's regrade rc 0).
  - ObjBounds 603528b8b has no reading yet. No devmk image goes to board 2 before its reading.

**Stages.** `spec/mk-eval-stages.sh` makes them with the 3 Nov stages' options:
- lbm, nab and imagick run both arms in one stage;
- xz, mcf and gcc get one stage per arm, and gcc is stripped.

On board 1 (bitstream 3, kit bkit12-perf-b3-b1, the 3 Nov kit), each benchmark runs dev then devmk, back to back
(links in spec/stages-eval-b1):

| Benchmark | Stage | Arms | Step / booking |
|---|---|---|---|
| lbm | both-arm | MapLife and Full | 5400 / 120 |
| nab | both-arm | MapLife and Full | 6000 / 130 |
| imagick | both-arm | MapLife and Full | 1200 / 45 |
| xz | per-arm | Full | 16200 / 310 |
| mcf | per-arm | Full | 30000 / 540 |
| gcc | per-arm | Full | 900 / 40 |

The Spatial arm (ObjBounds, board 2) follows in a later addendum.

**QEMU twin (done, kit-q34).** Every stage of both sets is rc 0, specdiff PASS, and its outputs are identical to the
baseline's twin (census-q34): 12/12 comparisons per set (census-eval-boarddev, census-eval-boarddevmk). The twin's
in-context instret for run commands, as ratios:

| Benchmark | dev / baseline | devmk / dev |
|---|---|---|
| lbm | 0.8521 | 0.9996 |
| nab | 0.9256 | 0.9952 |
| imagick | 0.9279 | 0.9696 |
| xz | 0.9966 | 0.9904 |
| mcf | 0.8168 | 0.9379 |
| gcc | 0.9475 / 0.9477 (Spatial / Full) | 0.9678 / 0.9677 |

- The compiler lane attributes mcf's dev drop to two of the external collaborator's commits:
  - f9d76538e477, leaf functions without a frame, ~90 %;
  - ac15af37bfa8, 32-bit results extended by the callee, ~10 %.
- The compiler lane's C4 shares (rewrite pairs / in-context instret) are: mcf 6.21 %, gcc 2.59 %, imagick 1.87 %,
  xz 0.84 %, nab 0.48 %, lbm 0.043 %. They predict devmk's drop. Measured, devmk's drop exceeds the share on imagick
  (3.0 %) and gcc (3.2 %), and matches it on mcf, nab and lbm.

**Baseline on silicon** (bitstream 3, 3 Nov; run commands, in-context cycles / instret):

| Benchmark | Arm | Cycles | Instret |
|---|---|---|---|
| lbm | MapLife | 36,945,072,184 | 7,281,772,675 |
| lbm | Full | 36,958,233,429 | 7,281,767,823 |
| nab | MapLife | 24,263,104,679 | 7,993,075,253 |
| nab | Full | 24,602,210,442 | 7,993,255,393 |
| imagick | MapLife | 402,555,914 | 108,724,600 |
| imagick | Full | 395,108,674 | 108,744,930 |
| xz | Full | 142,418,790,940 | 50,837,316,422 |
| mcf | Full | 312,411,004,637 | 35,326,234,070 |
| gcc | Full | 205,767,280 | 30,690,680 |

**Predictions.**

| # | Prediction |
|---|---|
| E1 | Every boot's control passes. Every arm is rc 0 and specdiff PASS, with outputs equal to the twin's. |
| E2 | On the board, instret ratios dev / baseline and devmk / dev equal the twin ratios above within 0.3 % (board instret has matched the twin to 0.02 % per image). |
| E3 | Step 1, cycles: dev / baseline < 1 on lbm, nab, imagick, mcf and gcc, the five whose instructions drop by > 5 %. xz (−0.3 % instructions) is reported, not predicted. |
| E4 | Step 2, cycles: devmk / dev < 1 on mcf, gcc and imagick. Each removed pair is a capability store plus a capability load, so the compiler lane's hypothesis is that cycles fall by at least the instruction share. E4b, which tests that hypothesis: on mcf, 1 − cycles ratio ≥ 1 − instret ratio (≥ 6.2 %). |
| E5 | Outside time per command (run time − in-context cycles / 25 MHz) changes by less than in-context time does. No magnitude is predicted, because the compiler does not touch the kernel. |

**Not predicted.** No magnitude is given for E3. The 3 Nov profiles say the cycle effect depends on where the removed
instructions sit: leaf prologues, sign extensions, the comparator calls.

**What would refute what.**
- **E2 fails.** The board ran different code from the twin. This is an instrument problem; no step's verdict stands
  until it is explained.
- **E3 or E4 fails on a benchmark whose instructions fell.** On that benchmark, the removed instructions were not on
  the critical path, or the new code costs more cycles per instruction. Both are reported as findings.

**Limits.**
- N = 1 per arm. The run-to-run spread is known only for imagick (the 3 Nov r10 stages, spec/census-repeat and
  board out-hfl/538.imagick_r-test-r10-*); it is not restated here.
- The board ran the baseline on 1-3 Nov and runs these later; board-to-board and day-to-day drift is unmeasured
  except by the known-good control.
