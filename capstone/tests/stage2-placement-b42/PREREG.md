# Code placement alone on b42c8c644: LSHOFFL relinked three ways, with only its layout changed (pre-registration, 2026-11-27)

**Why.** The LSRH S bracket (tests/spec8-lsrh-board RESULTS, e09ecb0844a4) read perlbench +3.55 s (+1.27 %) and nab
+2.24 s (+0.31 %). Its twin difference was -34 instructions and -0.042 %.
- LSRH resizes 18 of perlbench's 4,761 text functions, and moves 4,717.
- On b42c8c644 the I$ and D$ are 32 KiB, 8 ways, 16 B lines (core/include/capstone_cv64a6_imafdc_sv39_config_pkg.sv
  :93-98), so 4 KiB per way. The set index lies inside the page offset, so a relink moves code and data across cache
  sets and lines. The branch predictors index on PC bits too.
- ZCBB and every stage-2 CRVC image move nearly every function in the same way: ZCBB moves 4,713 of perlbench's
  functions against LSHOFFL, and resizes 720.

**This measures how far a board time moves when only placement changes.** It is the control that the ZCB pair
(stage2-zcb-b42, Amendment 1) and stage 2's cycle band are to be graded against.

Written before any perturbed image was built or booted.
- One tool check ran first: relinking LSHOFFL's perlbench objects in their recorded order, with its own link line,
  reproduces LSHOFFL's perlbench_r.full byte for byte (sha256 4ce92612b4d8f9cc). So the relink path itself changes
  nothing.

## Arms

All arms are LSHOFFL's own object files: llvm-sinkland clang 35580b17947b74ec, lld c45af5acebd278ec, B5TL1's flags.
They are relinked by each benchmark's recorded link line (link.txt) through the arm's capstone-cc. No object is
recompiled. Only the four board natives are relinked: cpugcc_r, imagick_r, perlbench_r and nab_r.

| arm | the change to the link line |
|---|---|
| LSHOFFL | none: the existing images (S-LSHOFFL, fw_payload sha256 21d18193b86359a8) |
| PL1 | the benchmark's .o inputs permuted by Python's random.Random(1).shuffle |
| PL2 | the same, with seed 2 |
| PL3 | the recorded order, with a pad object first: one section .text.zpad of 1,040 B (65 × 16 B), held live by --undefined |

- **What moves:**
  - the runtime archive stays first, so _start is unchanged;
  - the benchmark's code, its rodata, libc and builtins all move.
  - PL1 and PL2 reorder whole objects: functions keep their order within an object.
  - PL3 shifts everything after the runtime by 1,040 B. That is 1,040 mod 4,096 within the page, which is not a
    multiple of 32 or 64.
- **The images** are built by c4/spec8/board/make-board.sh, exactly as S-LSHOFFL was. That is the same four campaign
  groups and stages, the same init, and the same kernel, firmware and FW_FAST_RDTIME. Only bin/native/* differs. The
  script's overlay check must read 0 mismatches. Each image's sha256 is recorded in an amendment before its first boot.

## The twin check: is each relink placement and nothing else?

- **One twin boot per PL arm,** on spec8's init restricted to the four board stages (STAGES=gcc, imagick, perlbench,
  nab). Each boot runs under the QEMU lock, with the trap twin capstone-qemu c42ff84d0799 (sha256 457477d73dde1a2d),
  no CRVC or Zcb properties, at shift 1.
- **The readers:**
  - c4's count8s4 reader for instructions and output hashes;
  - fetch8 (perf/cclever/fetchbytes.py per native window) for fetch bytes.
- **The reference** is S4-LSHOFFL's reading of the same four natives on the same binary at shift 1:
  - instructions: gcc 21,710,898, imagick 94,485,851, perlbench 3,094,620,533, nab 7,154,811,027;
  - outputs as recorded there.
- **The rule:** a PL arm is a placement control for a stage only if two things hold:
  - its twin count equals LSHOFFL's within ±0.001 % per stage;
  - its outputs are equal.
  - Address-dependent relaxation could change a few instructions; the band allows for that and nothing more.
  - A PL arm that fails is dropped for that stage, and the board reading of it is not used.
- **Prediction:** every arm passes. Exact equality is predicted for PL1 and PL2, since no call-relaxation choice
  should depend on object order alone. PL3 too.

## The board session

- **Board 2, b42c8c644, no reflash, five boots, back to back:** LSHOFFL, PL1, PL2, PL3, LSHOFFL2.
  - A reflash, or another bitstream's boot, inside the five voids the session.
  - Preflight runs before each boot and its exit line is kept, as in the LSRH S bracket. The images are the S
    images' size, 25,856,520 bytes, proven on this seal by S-LSHOFF.
- It runs when the board lane and runtime give it a slot, after the ZCB pair's six boots.
- The board needs no new seal.

## The reading

- t is up1 - up0 on each stage's @@SPEC-END run line (0.01 s).
- **Per stage s:**
  - placement spread P_s = max - min of t over LSHOFFL, PL1, PL2 and PL3, using the arms that pass the twin check;
  - boot spread B_s = |t(LSHOFFL2) - t(LSHOFFL)|.
- **Placement is resolved on stage s if P_s > B_s + 0.02 s.** Otherwise it is not resolved above boot-to-boot noise
  in this session.
- **The drift control** is imagick's val phase, the same plain binary in all five boots. It fires if any val reading
  lies outside [min - q, max + q], with min and max the two LSHOFFL boots' and q = 0.0005 × max + 0.02 s. If it fires,
  every stage is UNDECIDED.
- **UNDECIDED** also covers a stage that faults, is cut, fails specdiff, or has output hashes that differ from
  LSHOFFL's.

## What the reading is used for (fixed here, before it exists)

1. **LSRH, after the fact:** its excess on stage s is E_s = t(LSRH) - t(LSHOFF) in its own bracket: perlbench
   3.55 s, nab 2.24 s.
   - **Within placement** if E_s ≤ P_s + B_s + 0.02 s. LSRH's failure on that stage is then recorded as
     indistinguishable from a placement effect.
   - **Not within placement** otherwise. The failure stands as LSRH's own, beyond what these relinks show.
   - LSRH's verdict, "does not enter", is not reopened either way. It is a one-sided gate, and it failed as written.
2. **The ZCB pair:**
   - its gain or failure on a stage is attributed to Zcb only if |mean(ON) - mean(OFF)| > P_s + range(OFF) + 0.02 s;
   - otherwise it is recorded as indistinguishable from placement.
3. **Stage 2:**
   - in the go/no-go, the "spread" of every stage becomes max(P_s, its in-session boot spread);
   - a gain counts toward the ≥ 1 % go threshold only beyond P_s.
   - That pre-registration cites this one.

## Predictions

The question this answers is open, so these are stated without strong priors.

1. **Every PL arm passes the twin check** (see above).
2. **perlbench:**
   - P_perl is predicted to be resolved, that is P > B + 0.02 s.
   - Its size is not predicted. If P_perl ≥ 3.55 - B - 0.02 s, LSRH's perlbench failure is within placement.
3. **gcc and imagick:** P is predicted unresolved, at most 0.03 s.
   - They are short, at 5.6 s and 11.2 s.
   - On this board, imagick read -0.05 s under LSRH, whose twin count was -1.77 %.
4. **nab:** no prediction.
   - Its two LSHOFF boots agreed to 0.05 s, so B_nab is small.
   - Any P_nab above about 0.1 s is resolved.

## What this does not decide

- **Whether placement is the mechanism of LSRH's slowdown.** Three relinks sample placement, they do not span it. A
  small P_s does not show that no placement would reproduce E_s.
- Which cache, predictor or TLB effect carries any P_s.
- P1, x264, mcf, xz and lbm.
