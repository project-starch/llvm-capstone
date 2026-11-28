# Code placement alone on b42c8c644, LSHOFFL against five relinks of its own objects: results (2026-11-27)

Pre-registration: PREREG.md, f4a6996c72c9, with Amendments 1-3: 97f4a5f07f3f (two pad arms, PL1/PL2 predicted to
fail the twin), 4758c2241785 (the images) and 1c159be19da5 (the twin verdicts and the boot list). Each was pushed
before the first board boot.

**Placement alone moves perlbench by 11.38 s (4.1 %) and nab by 2.21 s, both resolved above boot-to-boot spread.**
- perlbench: LSHOFFL against LSHOFFL2 is B = 1.58 s.
- nab: B = 0.45 s.
- imagick resolves by one tick, at 0.05 s. gcc does not move: 0.00 s.
- The drift control holds, and every stage of every boot passes specdiff with outputs equal to LSHOFFL's.

**What that decides, by the pre-registered uses:**
- **LSRH:** its board failure on perlbench (+3.55 s) and nab (+2.24 s) is **within placement** on both stages.
- **The ZCB pair:** its failure (perlbench +1.15 s, nab +1.00 s) is **indistinguishable from placement** on every
  stage, so it is not attributed to Zcb. The "does not enter" verdicts of both stand.
- **Stage 2's spreads** are now numbers: perlbench's P is 11.38 s, so a perlbench GO needs a saving of at least about
  14.2 s.

## The session

place3-b2-20261126T2207 on board 2, b42c8c644. The board lane ran it, and the bitstream was read back before and
after every boot. There was no reflash and no other bitstream's boot inside the seven.
- The boot list is Amendment 3's: LSHOFFL, PL1, PL2, PL3, PL4, PL5, LSHOFFL2.
- This lane read every number below from the uart captures, after checking each capture's sha256 against the
  board lane's row.

| boot | image (payload sha256) | uart sha256 | RUN END |
|---|---|---|---|
| LSHOFFL | S-LSHOFFL 21d18193b86359a8 | 22c18d8d0d0616f4 | rc=0 01:13:23Z |
| PL1 | S-PL1 684b81534935abf3 | 875d4220fe8ed849 | rc=0 01:41:02Z |
| PL2 | S-PL2 24e1097ad75d0da0 | cf7bd84496e520ac | rc=0 02:08:39Z |
| PL3 | S-PL3 136615641b452558 | ee9f844f04d6c3e3 | rc=0 02:47:30Z (the rerun) |
| PL4 | S-PL4 1919e53e73562558 | a7238da64dba52de | rc=0 03:15:17Z |
| PL5 | S-PL5 c7ff79012d367c16 | cbbc9c64279b584d | rc=0 03:43:07Z |
| LSHOFFL2 | S-LSHOFFL 21d18193b86359a8 | 83bb1793af6879e7 | rc=0 04:10:49Z |

- **Each payload's sha equals Amendment 2's,** gated before its boot. smoke.dom read rc=7, as predicted, in every
  boot.
- **Preflight:** each boot's run log carries C10's "proceeded by runtime's ruling; size proven on this seal".
- **PL3's first attempt** produced no reading, and so is not one of the seven.
  - The JTAG load's GDB step timed out ("Timed out after 120s waiting for busy to go low"). It stopped before the
    continue, and no UART byte came.
  - The readback before and after named b42c8c644.
  - It is not a reflash and not another bitstream's boot, so the session is not voided. The runner then ran its one
    rerun, which read.
- **Graders:** v2 passes all four stages in every boot, and v1 agrees. Each capture has 0 "sublet: fault" lines.
- **Outputs:** this lane compared each boot's four hash blocks (5, 8, 557 and 5 lines) with LSHOFFL's, sorted, with
  interleaved tick lines removed. All six are equal. Altering one line of the reference makes the comparison fire.

## The reading

t is up1 - up0 on each stage's @@SPEC-END run line, in seconds. Each P_s is taken over LSHOFFL and the arms that
passed the twin rule on that stage (Amendment 3):
- nab: LSHOFFL and PL1-PL5;
- perlbench: LSHOFFL, PL2, PL3, PL4, PL5;
- gcc and imagick: LSHOFFL, PL3, PL4, PL5.

| stage | LSHOFFL | PL1 | PL2 | PL3 | PL4 | PL5 | LSHOFFL2 | P_s | B_s | resolved (P > B + 0.02) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gcc | 5.55 | (5.57) | (5.56) | 5.55 | 5.55 | 5.55 | 5.55 | 0.00 | 0.00 | no |
| imagick | 11.24 | (11.27) | (11.26) | 11.19 | 11.21 | 11.20 | 11.22 | 0.05 | 0.02 | **yes**, by one tick |
| perlbench | 279.16 | (272.21) | 269.79 | 277.70 | 281.17 | 279.98 | 277.58 | **11.38** | 1.58 | **yes** |
| nab | 727.50 | 729.64 | 727.84 | 727.43 | 727.57 | 727.92 | 727.95 | **2.21** | 0.45 | **yes** |

Parenthesised readings are from stages the arm failed the twin rule on. They are booted but not read.

- **The drift control holds.** imagick's val reads 5.43 in the first six boots and 5.42 in LSHOFFL2. The window is
  [5.397, 5.453].
- **The extremes:**
  - **perlbench:** the extreme pair is PL2 (269.79) and PL4 (281.17). Over LSHOFFL and the three pure pad arms alone,
    the spread is 3.47 s (PL3 277.70 to PL4 281.17).
  - PL2 passes the twin rule there (+8,161 instructions, 0.00026 %), but it resizes 894 of perlbench's functions
    through linker relaxation (Amendment 1). So its 9.37 s below LSHOFFL may carry code effects as well as placement.
    **The pre-registered P_s is 11.38 s.** The 3.47 s is reported as a sensitivity, not a substitute.
  - **nab:** PL1 is the extreme, at +2.14 s, on a nab whose twin count differs by 28 instructions.
- **LSHOFFL's two readings differ more than in earlier brackets:** perlbench by 1.58 s and nab by 0.45 s. In the ZCB
  pair's three OFF boots of the same image the ranges were 0.56 s and 0.08 s.

## Predictions

1. **Every PL arm passes the twin check:** already graded in Amendment 3. PL3, PL4 and PL5 pass on all stages. PL1
   passes only nab, and PL2 passes nab and perlbench; Amendment 1's prediction for PL2 is refuted on perlbench.
2. **P_perl resolved: HOLDS.** It read 11.38 s against 1.58 + 0.02. It is ≥ 3.55 - B - 0.02 = 1.95 s, so LSRH's
   perlbench failure is within placement, as the prediction's conditional said.
3. **gcc and imagick unresolved, at most 0.03 s:**
   - **HOLDS on gcc** (0.00 s).
   - **REFUTED on imagick.** It read 0.05 s, resolved by one tick: PL3's 11.19 against LSHOFFL's 11.24.
4. **nab:** no prediction was made. It read P = 2.21 s, resolved.

## The fixed uses

**1. LSRH, after the fact** (tests/spec8-lsrh-board RESULTS, e09ecb0844a4):
- perlbench: E = 3.55 ≤ P + B + 0.02 = 12.98, **within placement**. It is also within with the pads-only sensitivity:
  3.47 + 1.58 + 0.02 = 5.07.
- nab: E = 2.24 ≤ 2.21 + 0.45 + 0.02 = 2.68, **within placement**. The margin is 0.44 s.
- LSRH's failure on both stages is recorded as indistinguishable from a placement effect. Its verdict, "does not
  enter", is not reopened: it was a one-sided gate, and it failed as written.

**2. The ZCB pair** (tests/stage2-zcb-b42 RESULTS, d0c5af96e033): attributed to Zcb only if |Δ| > P_s + range(OFF) +
0.02.

| stage | ZCB pair Δ | limit | reading |
|---|---:|---:|---|
| perlbench | +1.153 | 11.96 | indistinguishable from placement |
| nab | +0.997 | 2.31 | indistinguishable from placement |
| gcc | -0.013 | 0.02 | indistinguishable from placement |
| imagick | -0.010 | 0.07 | indistinguishable from placement |

**3. Stage 2** (tests/stage2-crvc-board, 4d0f6b69116f and amendments):
- In its go/no-go, every stage's spread becomes max(P_s, its in-session range of OFF). The P_s are perlbench 11.38 s,
  nab 2.21 s, imagick 0.05 s and gcc 0.00 s.
- A gain counts toward the 1 % only beyond P_s.
- **perlbench:** a GO needs -Δ ≥ 11.38 + 0.01 × mean(OFF), about 14.2 s, or 5.1 % of 278 s. That is r ≥ 0.29 of its
  17.306 % fetch saving, against the band's 0 to 0.37.
- **gcc:** P = 0, so the gain rule's separation (max(ON) < min(OFF) - 0.02 s) and -Δ ≥ 0.01 × mean(OFF) decide it.
  That is about 0.06 s, six ticks: r ≥ 0.12 of its 9.068 % fetch saving.
- **No-regression:** perlbench may read up to 11.38 + 0.0005 × t + 0.02 s worse before it fails, nab up to about 2.6 s.

## What this does not show

- **Which mechanism carries the 11.38 s.** It could be the I$, the D$, the branch predictors or the TLB. Five relinks
  sample placement, they do not span it. A larger P_s is possible, and no placement is shown to be worst.
- **Whether PL2's perlbench reading is placement only.** Its relaxation changes code in 894 functions while keeping
  the count within the twin band.
- P1, x264, mcf, xz and lbm.
