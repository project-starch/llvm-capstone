# Stage 2 step 4 on Line B, fetch bytes of CRVC, Zcb with a capability base, and both, on the trap twin: results (2026-11-27)

Pre-registration: PREREG.md, ebbf6d221cb0, written before any measuring boot.

**CRVC saves 17.3 % of perlbench makerand's fetch bytes and 9.1 % of gcc t1.c's.** The other six stages read P1's
step-4 figures to three digits.
- **Zcb with a capability base:** 0.07 to 1.29 % per stage.
- **The two together:** their sum, within 0.003 points on every stage.
- **Instructions:** every arm counts what LSHOFFL counts within 0.0085 %, with outputs EQUAL on all eight stages.
- **The predictions:**
  - prediction 1 (instructions) and prediction 3 (the FP-heavy stages) hold;
  - prediction 2 is refuted on perlbench;
  - prediction 4 is refuted on perlbench;
  - prediction 5's floor clause is refuted on nab and lbm, and its sum clause holds everywhere.

These readings are the input to the cycle band, pre-registered from CRVCB's reading as tests/stage2-crvc-board
(4d0f6b69116f) before this RESULTS: CRVCZB is not a candidate there, since ZCB does not enter (tests/stage2-zcb-b42
RESULTS, d0c5af96e033).

## The boots

Four eight-stage boots, c4/spec8/S4-*.
- **Platform:** capstone-qemu c42ff84d0799 (sha256 457477d73dde1a2d), with capmix3 at shift 1. Each boot took the
  QEMU lock.

| boot | properties | ended | stages |
|---|---|---|---|
| S4-LSHOFFL | none | rc 0 after 8,280 s | 8/8, none cut |
| S4-CRVCB | x-capstone-crvc, x-capstone-crvc-protected | rc 0 after 8,126 s | 8/8, none cut |
| S4-ZCBB | zcb=true | rc 0 after 8,245 s | 8/8, none cut |
| S4-CRVCZB | all three | rc 0 after 8,172 s | 8/8, none cut |

- **No SIGILL in any boot.**
- **Each boot carries one sublet fault:** `.perl-native` cause=28 at pc 0x3ff7c01498, in the test.pl command of
  perlbench's stage. It is the same in all four arms and in spec8's LSHOFF and LSHON boots, so it predates step 4.
  That command is not a native window, and it is not read.
- **Order:** the procedure's order, LSHOFFL, CRVCB, ZCBB, CRVCZB, held.
  - CRVCZB was queued behind the placement control's five twin boots, so that that control's amendment landed before
    its board slot.
  - The wrapper that did it (c4/spec8/reorder-after.sh) logged ZCBB's end line five minutes early: its test read
    /proc cmdline with a space, where the separator is NUL.
  - The QEMU lock kept the next boot waiting until ZCBB ended. A correction line with ZCBB's true end is in the log.

## The readers

- **Instructions and outputs:** count8.py at shift 1 (c4's count8s4).
- **Fetch bytes:** perf/cclever/fetchbytes.py, sha256 24d05fb3bace9e7d, over each native command window with its own
  ELF.
  - Each window's load delta must equal its enters line's base minus 0x10000. That is independent of the tool's own
    search.
- **Repaired before any arm but LSHOFFL was read.**
  - The search checked its 2,000 hottest user PCs against the image, including PCs outside a small image, so lbm
    read "0 deltas fit". The check now applies only where a candidate delta puts the PC inside the image.
  - **Controls:** the old and new readers agree on nab's window 2 and gcc's window 15, and a wrong ELF exits 1.
- **perlbench is cross-checked by an independent reader.**
  - It takes instruction lengths from llvm-objdump -d and the delta from the enters line, with no search.
  - It agrees with fetchbytes to the byte on LSHOFFL and CRVCB (10,619,819,362 and 8,781,923,040).
  - With LSHOFFL's ELF under CRVCB's profile, it reads 1,236,996,935 instructions at non-starts, so the check fires.
- **perlbench is the makerand window,** the command the board times. The test.pl command runs through natperl, is not
  a native window, and is not counted.

## Instructions (against S4-LSHOFFL)

| stage | LSHOFFL | CRVCB | ZCBB | CRVCZB | outputs |
|---|---:|---:|---:|---:|---|
| lbm | 6,092,401,820 | 0 | 0 | 0 | EQUAL |
| mcf | 26,644,261,998 | -4 | -4 | -4 | EQUAL |
| nab | 7,154,811,027 | 0 | 0 | 0 | EQUAL |
| xz | 49,619,823,268 | -247,129 (-0.0005 %) | +12 | -247,203 (-0.0005 %) | EQUAL |
| gcc | 21,710,898 | -1,323 (-0.0061 %) | -24 | -1,840 (-0.0085 %) | EQUAL |
| imagick | 94,485,851 | -397 | -88 | -397 | EQUAL |
| x264 | 506,620,171,647 | -782,197 (-0.0002 %) | -1,255,109 (-0.0002 %) | -1,284,542 (-0.0003 %) | EQUAL |
| perlbench | 3,094,620,533 | -2,006 | -4 | -2,018 | EQUAL |

"Outputs" compares each arm's output hash sets with LSHOFFL's, per stage.

## Fetch bytes saved (1 - bytes(arm) / bytes(LSHOFFL), the benchmark's own image)

| stage | LSHOFFL bytes | CRVCB | ZCBB | CRVCZB | P1 CRVC | P1 ZCB |
|---|---:|---:|---:|---:|---:|---:|
| perlbench (makerand) | 10,619,819,362 | **17.306 %** | 0.072 % | 17.378 % | 8.467 % (test-sw) | 0.889 % (test-sw) |
| gcc (t1.c) | 77,255,052 | **9.068 %** | 0.380 % | 9.450 % | 14.879 % (train01) | 0.593 % (train01) |
| x264 | 1,857,395,105,930 | 2.162 % | 1.293 % | 3.455 % | 2.155 % | 1.289 % |
| mcf | 87,378,218,004 | 6.613 % | 0.081 % | 6.694 % | 6.613 % | 0.081 % |
| xz | 173,345,885,300 | 1.859 % | 1.259 % | 3.119 % | 1.859 % | 1.259 % |
| imagick | 351,237,504 | 0.379 % | 0.992 % | 1.371 % | 0.375 % | 0.982 % |
| nab | 25,574,707,108 | -3.532 % | 0.220 % | -3.312 % | -3.532 % | 0.220 % |
| lbm | 23,242,341,846 | -2.982 % | 0.000 % | -2.982 % | -2.982 % | 0.000 % |

**User instructions outside the image:** perlbench makerand 61,982 in every arm, so the uncompressed libc and runtime
carry almost none of its fetch. They are reported, never folded in.

## Predictions

1. **Each arm within 0.05 % of LSHOFFL per stage, outputs equal in all four: HOLDS.** The largest difference is gcc
   under CRVCZB, at -0.0085 %.
2. **CRVCB within ±25 % (relative) of P1 on perlbench, x264, mcf and xz; gcc above 5 %:**
   - **REFUTED on perlbench.** It read 17.306 %, against a band of 6.35 to 10.59 %.
   - **HOLDS** on x264 (2.162 %), mcf (6.613 %) and xz (1.859 %), and on gcc (9.068 % > 5 %).
3. **nab and lbm negative, imagick between -1 and +2 %: HOLDS.** They read -3.532 %, -2.982 % and +0.379 %.
4. **ZCBB within ±0.3 points of P1 on every stage but gcc; gcc between 0 and 1.5 %; none below -0.02 %:**
   - **REFUTED on perlbench.** It read 0.072 % against P1's 0.889 %, 0.817 points below.
   - **HOLDS** on the other six stages (the largest difference is imagick's +0.010 points), on gcc (0.380 %), and on
     the floor: the lowest reading is lbm's 0.000 %.
5. **CRVCZB:**
   - **"At least the larger of CRVCB's and ZCBB's, minus 0.1 points": REFUTED on nab and lbm.** nab reads -3.312 %
     against a floor of 0.120, and lbm -2.982 % against -0.100. The clause did not allow for a negative CRVC saving:
     there the combination sits below ZCB alone, by exactly CRVC's loss.
   - It holds on the other six stages.
   - **"Their sum within 0.3 points": HOLDS on all eight stages,** the largest gap being gcc's +0.003. The two features
     compress disjoint instruction classes, as predicted, and this is the first measurement of the combination.

**Why perlbench departs from P1 on both features:**
- **The workloads differ.** P1's row was 500.perlbench_r-test-sw; Line B's stage is makerand. gcc's rows differ the
  same way (train01 against t1.c), and the pre-registration excluded gcc from prediction 2 for that reason.
- **The six stages that share P1's workloads match P1** to three digits on both features, so the twin, the arms and
  the reader agree with P1's.
- Why makerand compresses about twice as well under CRVC, and about a twelfth as well under Zcb, is not identified.
  No census of the instruction classes is claimed.

## What it decides

- **These are the cycle band's inputs.** The band and the go/no-go are pre-registered in tests/stage2-crvc-board
  (4d0f6b69116f, amended at f27d0ba8979f and ade77cfc5ca3), from CRVCB's perlbench and gcc readings above.
- **Per-image choice under option B:** CRVC's saving is negative on nab and lbm, so those ship without +cap-crvc. The
  board's ON image carries LSHOFFL's own nab_r (CRVCBN).
- **ZCB:** ZCBB's twin saving on perlbench is 0.072 %. The board read the ZCBB image as slower (+0.41 %), and ZCB
  does not enter (d0c5af96e033). CRVCZB therefore goes to no board.

## What this does not show

- Cycles, I$ misses and silicon. The board pair of tests/stage2-crvc-board reads those.
- A libc and runtime rebuilt with the features. On makerand they carry about 62 thousand of 3.1 billion user
  instructions.
- Which instruction classes make up makerand's 17.3 %.
