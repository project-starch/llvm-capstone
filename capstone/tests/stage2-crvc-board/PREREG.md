# Stage 2 on Line B: the CRVC cycle band and the go/no-go for a stage-2 seal (pre-registration, 2026-11-27)

The plan is plans/2026-11-25-stage2-trap-route-decoder-spec.md, section (d) of the Line B part, with the mode's home
converged with rtl (bit 0 of the srevroot write). Written after step 4's CRVCB twin reading, before any synthesis of
the stage-2 RTL (capstone-ariane stage2-crvc 796af23d9), and before the placement control's board session.

**It decides two things:**
- whether stage 2 is worth a seal: rtl's synthesis-only gate, then a full build, may proceed on it;
- what a board pair on that seal must read for stage 2 to be proposed to the lead.

It cites:
- tests/stage2-lineb-step4 (PREREG ebbf6d221cb0) for the fetch readings;
- tests/stage2-placement-b42 (f4a6996c72c9, Amendments 1 and 2: 97f4a5f07f3f, 4758c2241785) for P_s;
- the two board results that showed a relink alone moving perlbench on b42c8c644:
  - LSRH, tests/spec8-lsrh-board RESULTS e09ecb0844a4 on compiler/imagick-loop-shapes: FAIL, perlbench +1.27 %, nab
    +0.31 %;
  - ZCB, tests/stage2-zcb-b42 RESULTS d0c5af96e033: the ZCBB image FAILS, perlbench +0.41 %, nab +0.14 %.

## The image: CRVCB, and only CRVCB

- **CRVCZB is not a candidate.** ZCB does not enter (d0c5af96e033), and the placement control cannot reopen that
  verdict: its use there (Amendment 1 of stage2-zcb-b42) decides attribution, not entry. Step 4's ZCBB and CRVCZB
  readings are graded in step 4's RESULTS and enter nothing here.
- **Per-image choice under option B.** CRVCB's fetch saving is negative on nab (-3.532 %) and lbm (-2.982 %), so both
  ship without +cap-crvc.
  - On the board stages that means the ON image carries CRVCB's perlbench_r, cpugcc_r and imagick_r natives, and
    **LSHOFFL's own nab_r, byte-identical**.
  - So nab is an in-session control: its binary is the same in both images. Its band is [0, 0].
  - imagick (+0.379 %) is positive, so it ships with the feature. Its band lies below one 0.01 s tick.

## The input: step 4's CRVCB reading (twin, shift 1)

The reading is the c4 spec8 boots S4-LSHOFFL and S4-CRVCB, on capstone-qemu c42ff84d0799 (sha256 457477d73dde1a2d),
under step 4's PREREG.
- Fetch bytes come from perf/cclever/fetchbytes.py (sha256 24d05fb3bace9e7d), per native command window, summed per
  stage.
- Each window's load delta must equal its enters line's base minus 0x10000.

| stage | LSHOFFL insns | CRVCB insns | diff | LSHOFFL bytes | CRVCB bytes | B/insn | fetch saved |
|---|---:|---:|---:|---:|---:|---|---:|
| perlbench (makerand) | 3,094,620,533 | 3,094,618,527 | -2,006 | 10,619,819,362 | 8,781,923,040 | 3.432 → 2.838 | **17.306 %** |
| gcc (t1.c) | 21,710,898 | 21,709,575 | -1,323 (-0.006 %) | 77,255,052 | 70,249,950 | 3.558 → 3.236 | **9.068 %** |
| imagick | 94,485,851 | 94,485,454 | -397 | 351,237,504 | 349,907,162 | 3.717 → 3.703 | 0.379 % |
| nab | 7,154,811,027 | 7,154,811,027 | 0 | 25,574,707,108 | 26,478,095,328 | 3.575 → 3.701 | -3.532 % |
| x264 | 506,620,171,647 | 506,619,389,450 | -782,197 | 1,857,395,105,930 | 1,817,243,110,776 | | 2.162 % |
| mcf | 26,644,261,998 | 26,644,261,994 | -4 | 87,378,218,004 | 81,600,027,966 | | 6.613 % |
| xz | 49,619,823,268 | 49,619,576,139 | -247,129 | 173,345,885,300 | 170,122,591,336 | | 1.859 % |
| lbm | 6,092,401,820 | 6,092,401,820 | 0 | 23,242,341,846 | 23,935,411,038 | | -2.982 % |

- **Outputs:** EQUAL on all eight stages. Every instruction difference is within 0.006 %, which is step 4's
  prediction 1, so the twin gate for the pair holds.
- **perlbench is the makerand window alone,** the command the board times. The stage's other command (test.pl
  through natperl) is not a native window and is not counted.
  - User instructions outside the image in that window: 61,982 in both arms. The uncompressed libc and runtime
    therefore carry almost none of makerand's fetch.
  - Kernel instructions are not counted.
- **The reader was repaired before this reading.** On a small image, the search's 2,000-PC check had counted user
  PCs outside the image, and it read "0 deltas fit" on lbm. The check now applies only to PCs that a candidate
  delta puts inside the image.
  - Controls: the old and new readers agree on nab window 2 and gcc window 15, and a wrong ELF exits 1.
- **perlbench, cross-checked by an independent reader.** Instruction lengths came from llvm-objdump -d (the bytes it
  prints for each instruction), and the delta from the enters line, with no search.
  - It agrees with fetchbytes to the byte on both arms: 10,619,819,362 and 8,781,923,040.
  - With LSHOFFL's ELF under CRVCB's profile, it reads 1,236,996,935 instructions at non-starts, so the check fires.

**Open risk: the perlbench reading is twice P1's.** Step 4 predicted CRVCB within ±25 % of P1's 8.47 %, and that
prediction is refuted on perlbench.
- **The workloads differ:**
  - P1's row was 500.perlbench_r-test-sw, not makerand;
  - gcc's row was train01, not t1.c, which step 4 already excluded for that reason.
- **The six stages that share P1's workloads read P1's figures to three digits:** x264 2.162 against 2.155, mcf
  6.613 against 6.613, xz 1.859 against 1.859, imagick 0.379 against 0.375, nab -3.532 against -3.532, lbm -2.982
  against -2.982.
- Why makerand compresses about twice as well is not identified, so no per-class census is claimed. The figure is
  the campaign's own flags on the board's own workload, so it is the right input either way.

## The band

time saved_s = fetch saved_s × r_s, with r_s in [0, 0.37] (the plan; ZCB9 on P1 silicon, N=1). Scale is the three
S-LSHOFFL boots of the ZCB pair on b42c8c644 (d0c5af96e033): perlbench 278.38 s, gcc 5.56 s, imagick 11.23 s, nab
727.50 s. The plan's 275.70 s for perlbench has no source found on this branch, and it is not used.

| stage | fetch saved | band | at scale | 1 % of OFF |
|---|---:|---:|---:|---:|
| perlbench | 17.306 % | 0 to -6.403 % | 0 to -17.83 s | 2.78 s |
| gcc | 9.068 % | 0 to -3.355 % | 0 to -0.187 s | 0.056 s |
| imagick | 0.379 % | 0 to -0.140 % | 0 to -0.016 s | |
| nab | 0 (LSHOFFL's binary) | 0 | 0 | |

- **The 0.37 was calibrated at 0.889 % fetch saved.** 17.3 % is about 20 times further out, and nothing says r is
  constant across that range.
- **The decision quantity is r_s, measured:** r_s = (-Δ_s / mean(OFF_s)) / fetch saved_s, with Δ_s = mean(ON) -
  mean(OFF) on the new seal.

## The board, after the placement session's RESULTS

**The seal N:** b42c8c644 + the CRVC decoder + the mode (rtl's stage2-crvc, or its successor), sealed.
- rtl's synthesis-only gate and lint numbers come first, as CLAUDE.md requires for a change on a loop cone.
- Its readback name is recorded in an amendment before the first boot.

**The kernel K:** trapk's final tree with the patch in the plan's "The mode's home" section.
- The patch is the binfmt.h bind of table_pa | crvc, plus the boot probe.
- Its twin checks (plan (b) item 3: a flagged image runs in the mode; the leak mutant trips the guard; with the guard
  also off, an ordinary process still decodes standard RVC) pass before any board image is built.

**The images**, both on K and built by c4/spec8/board/make-board.sh as S-LSHOFFL was:
- S-LSHOFFL-K: S-LSHOFFL's natives;
- S-CRVCB-K: CRVCB's perlbench_r, cpugcc_r and imagick_r, with LSHOFFL's nab_r.
- They differ only in the three CRVC natives.
- Their sha256 go in an amendment before the first boot.

**The boots, on board 2:**
1. **K0:** S-LSHOFFL-K on b42c8c644, before the reflash. This is the seal control's reference.
2. The reflash to N, read back.
3. **Six alternating boots on N:** OFF1, ON1, OFF2, ON2, OFF3, ON3 (OFF = S-LSHOFFL-K, ON = S-CRVCB-K), as in the
   ZCB pair. No reflash and no other bitstream's boot inside them.
- **Stages:** the S bracket's four, with t = up1 - up0 on each stage's @@SPEC-END run line.
- **This session does not boot before tests/stage2-placement-b42's RESULTS is pushed,** because every rule below
  reads P_s from it.

## What decides (fixed here, before any reading)

- **The inputs:** P_s is the placement spread on stage s from tests/stage2-placement-b42's RESULTS.
  spread_s = max(P_s, range(OFF_s)) over the three OFF boots on N.
- **A stage is UNDECIDED** if any of these holds:
  - that RESULTS leaves P_s UNDECIDED on it;
  - the drift control fires;
  - the stage faults, is cut, or fails specdiff;
  - its output hashes differ between boots.

**GO for proposing stage 2 to the lead requires all three:**
1. **A gain on perlbench or gcc, beyond placement:**
   - max(ON) < min(OFF) - 0.02 s;
   - and -Δ_s - P_s ≥ 0.01 × mean(OFF_s).
   - Only the part of the saving beyond P_s counts toward the 1 %.
2. **No stage worse:** on every decided stage, Δ_s ≤ 0.0005 × mean(OFF_s) + spread_s + 0.02 s (the LSRH board form,
   with spread_s in place of the boot spread).
3. **nab holds both ways,** its binary being identical: |Δ_nab| ≤ 0.0005 × mean(OFF_nab) + range(OFF_nab) + 0.02 s.
   - If it fails, the two images differ by more than their CRVC natives, and the session is UNDECIDED, not NO-GO.

**Otherwise NO-GO.** The stage-2 seal is then not proposed, and stage 2 stops on Line B.

**The drift control** is imagick's val phase, the same plain binary in all six boots on N. It fires if
|mean(val, ON) - mean(val, OFF)| > range(val, OFF) + 0.02 s, as in the ZCB pair.

**The seal control: seal_s = mean(OFF on N) - t(K0).**
- It is reported for every stage. It says "the seal costs time on s" if seal_s > 0.0005 × t(K0) + range(OFF_s) +
  0.02 s.
- It does not change the ON-against-OFF verdict, which is read within N.
- It goes to the lead beside it, because the campaign's other programs would pay it.
- N never replaces the campaign seal unless the verdict is GO, the seal control holds, AND the lead agrees.

## Predictions

At P_s = 0, for scale only, since P_s is not read yet: GO on perlbench needs r_perl ≥ 0.058 (2.80 s), and on gcc
r_gcc ≥ 0.150 (0.076 s). A P_s of a few seconds on perlbench, the size of LSRH's 3.55 s excess, would raise the
perlbench threshold to about r = 0.13.

1. **perlbench reads a saving (Δ < 0),** with r_perl in (0, 0.37].
   - At P1's perl calibration (0.37), it reads about -17.8 s, which is a GO.
   - At P1's xz calibration (0.05), it reads about -2.4 s, which is a NO-GO even at P_s = 0.
   - So the reading separates the two calibrations P1 left open.
2. **gcc reads flat or a saving** (Δ ≤ 0), no more than 0.19 s. Whether it clears r ≥ 0.150 is not predicted.
3. **imagick reads flat,** |Δ| < 0.03 s.
4. **nab reads flat** under rule 3.
5. **The seal control holds on every stage.** The decoder is out of every path a process without the flag takes.
6. **The predicted verdict is GO, carried by perlbench.** This is a prediction, not a band. The band itself rejects
   only r > 0.37 or a slowdown.

## What this does not decide

- **Which cache or front-end effect carries any saving.** I$ misses are not read.
- Whether the libc and runtime should be rebuilt with the feature. On makerand they carry almost no fetch.
- x264, mcf, xz and lbm on silicon: they are not in the S bracket.
- The choice between option A and option B, which stays with the lead.
