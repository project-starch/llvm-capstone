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

## Amendment 1 (2026-11-27, after the relinks were built and before any twin or board reading of them): two more pad arms

**The relinks, built by c4/placement/relink-pl.sh.** LSHOFFL's own exes and link lines were checked unchanged
afterwards. Sizes and changes against LSHOFFL:

| native | LSHOFFL .text | PL1 | PL2 | PL3 |
|---|---:|---:|---:|---:|
| perlbench_r | 1,865,584 | +9,484 | +292 | +1,040 |
| cpugcc_r | 8,144,164 | +27,000 | +39,008 | +1,052 |
| imagick_r | 1,772,332 | +12,620 | +14,716 | +1,056 |
| nab_r | 267,140 | -4 | +4 | +1,040 |

- **PL1 and PL2 are not placement alone.** They change the size of 1,073 and 894 of perlbench's 4,761 functions. The
  object shuffle changes the distances that RISC-V linker relaxation reads (calls, and the gp-relative forms), so it
  changes instructions.
  - Prediction 1's "exact equality for PL1 and PL2" is withdrawn before any reading.
  - They are now predicted to fail the twin rule (±0.001 %) on perlbench, gcc and imagick.
  - The twin rule stays as written, and decides.
- **PL3 shifts everything after the runtime by 1,040 B, and changes no function's size** on perlbench or nab. The
  extra 12 and 16 B on gcc and imagick are alignment between input sections.

**Added, so that the session keeps at least three placement controls if PL1 and PL2 fail:**
- **PL4:** as PL3, with a pad of 2,064 B (129 × 16).
- **PL5:** as PL3, with a pad of 3,088 B (193 × 16).
- Together with PL3 (1,040 B) and LSHOFFL (0), the in-page shifts are 0, 1,040, 2,064 and 3,088 B. They are spread
  across the page, and none is a multiple of 32.

**The session boots LSHOFFL first and LSHOFFL2 last. Between them, it boots exactly the PL arms that pass the twin
check on all four stages,** in the order PL1 to PL5. An arm that passes on only some stages is booted, and only its
passing stages enter P_s. P_s is computed over LSHOFFL and the booted arms as before.

## Amendment 2 (2026-11-27, before any twin or board reading of the PL arms): the board images

Built by c4/spec8/board/make-board.sh, the S-LSHOFFL script, from each PL arm's exes. Each overlay check reads: 574
files equal to source, 4 natives equal to the arm, 0 mismatches. Each payload is 25,856,520 bytes, keep=y,
fast-path-refs=2.

| image | fw_payload.bin sha256 |
|---|---|
| S-LSHOFFL (LSHOFFL and LSHOFFL2) | 21d18193b86359a88047d30036c0ef01407fa4c0805b2e794d2a7f781040c2cb (the ZCB pair's OFF image) |
| S-PL1 | 684b81534935abf36c2fff66c73d38d00d01831ebbe0df38bb74a3c3bde717a8 |
| S-PL2 | 24e1097ad75d0da0c96510d8715ca69e09b688b5a716b2674d5b48f59eddfdc7 |
| S-PL3 | 136615641b452558eb4066e2955eeb70e46e1b69dd6b751301c3b516db868f2d |
| S-PL4 | 1919e53e73562558b3fd2b4f7cf633b6965a0a6cb1efc9fe8aa626dc9c4672e0 |
| S-PL5 | c7ff79012d367c165e80ed66c20f6893ed7a4e343c1e36cf53834a9466b2022f |

- **The PL images and S-LSHOFFL differ only in the overlay's four natives.** The kernel Image and the firmware
  binary differ too, since the Image embeds the overlay and the firmware embeds the Image.
- **Which of them boot** is decided by the twin check alone, in an amendment pushed before the first board boot. That
  amendment gives the verdict per arm and per stage.

## Amendment 3 (2026-11-27, after the twin boots and before any board boot of a PL image): the twin verdicts and the boot list

**boot: LSHOFFL, PL1, PL2, PL3, PL4, PL5, LSHOFFL2**

This line is the session's boot list, in its order. Each listed arm boots its image with all four stages. The table
under "The verdicts" decides only which stages enter P_s.

| boot | image | fw_payload.bin sha256 (Amendment 2) | stages that enter P_s |
|---|---|---|---|
| LSHOFFL | S-LSHOFFL | 21d18193b86359a88047d30036c0ef01407fa4c0805b2e794d2a7f781040c2cb | all four |
| PL1 | S-PL1 | 684b81534935abf36c2fff66c73d38d00d01831ebbe0df38bb74a3c3bde717a8 | nab |
| PL2 | S-PL2 | 24e1097ad75d0da0c96510d8715ca69e09b688b5a716b2674d5b48f59eddfdc7 | nab, perlbench |
| PL3 | S-PL3 | 136615641b452558eb4066e2955eeb70e46e1b69dd6b751301c3b516db868f2d | all four |
| PL4 | S-PL4 | 1919e53e73562558b3fd2b4f7cf633b6965a0a6cb1efc9fe8aa626dc9c4672e0 | all four |
| PL5 | S-PL5 | c7ff79012d367c165e80ed66c20f6893ed7a4e343c1e36cf53834a9466b2022f | all four |
| LSHOFFL2 | S-LSHOFFL | 21d18193b86359a88047d30036c0ef01407fa4c0805b2e794d2a7f781040c2cb | B_s and the drift window only |

So P_s is taken over:
- nab: LSHOFFL, PL1, PL2, PL3, PL4 and PL5;
- perlbench: LSHOFFL, PL2, PL3, PL4 and PL5;
- gcc and imagick: LSHOFFL, PL3, PL4 and PL5.

**Unchanged:** B_s, the drift control (its window from the two LSHOFFL boots, now applied to all seven), UNDECIDED,
and every use under "What the reading is used for". A stage that fails the twin rule is booted with its arm but never
read.

### The twin boots

- **The boots:** c4/spec8/PLT1..PLT5, one per arm, on spec8's init with the four board stages (nab, gcc, imagick,
  perlbench).
- **The platform:** capstone-qemu c42ff84d0799 (sha256 457477d73dde1a2d), with no CRVC or Zcb property, capmix3 at
  shift 1, under the QEMU lock.
- **Every boot:** rc 0, 4/4 stages ended, none cut, no SIGILL.
- **The binaries are the board's.** Each PLT dir's four natives are byte-identical to its arm's strip.dom images,
  20 of 20. Those are the files Amendment 2's overlay check found in each S-PL payload.
- **Order:** they ran PLT3, PLT4, PLT5, PLT1, PLT2. They were moved ahead of step 4's last twin boot so that this
  amendment lands before the board slot. A twin count does not depend on boot order.
- **The reference** is S4-LSHOFFL, as pre-registered. The ±0.001 % band is, in instructions: nab 71,548, gcc 217,
  imagick 945, perlbench 30,946.
- **The one sublet fault in each boot** is the known `.perl-native` cause=28 at pc 0x3ff7c01498. It occurs in
  perlbench's test.pl command, outside the timed makerand window, and it is in S4-LSHOFFL too.

### The verdicts

Instructions are against S4-LSHOFFL (nab 7,154,811,027; gcc 21,710,898; imagick 94,485,851; perlbench
3,094,620,533).

| arm | nab | gcc | imagick | perlbench | outputs |
|---|---|---|---|---|---|
| PL1 | -28: **pass** | -14,086 (-0.065 %): fail | +21,645 (+0.023 %): fail | +3,841,211 (+0.124 %): fail | EQUAL, 4/4 |
| PL2 | -311: **pass** | +2,321 (+0.011 %): fail | +17,178 (+0.018 %): fail | +8,161 (+0.00026 %): **pass** | EQUAL, 4/4 |
| PL3 | 0: **pass** | 0: **pass** | -178 (-0.00019 %): **pass** | 0: **pass** | EQUAL, 4/4 |
| PL4 | 0: **pass** | 0: **pass** | -178: **pass** | 0: **pass** | EQUAL, 4/4 |
| PL5 | 0: **pass** | 0: **pass** | -178: **pass** | 0: **pass** | EQUAL, 4/4 |

- **imagick's -178 is the same in all three pad arms.** Step 4's control 1 read exactly that count for LSHOFFL
  itself on this binary (94,485,673, at shift 4). So it is S4-LSHOFFL's reading that lies 178 above. That is inside
  the band, and it is not attributed.
- **PL3's fetch bytes** (fetch8, the pre-registered reader):
  - equal to LSHOFFL's on nab and perlbench;
  - gcc -2 B, which is the 12 B of inter-section alignment that Amendment 1 noted;
  - imagick -712 B, which is its 178 instructions.

### Predictions graded

- **PL3, PL4 and PL5 pass on every stage: HOLDS.**
- **PL1 fails on perlbench, gcc and imagick: HOLDS.** It passes nab.
- **PL2 fails on perlbench, gcc and imagick: REFUTED on perlbench.**
  - It read +8,161 instructions there, 0.00026 %, inside the band. It holds on gcc and imagick, and PL2 passes nab.
  - PL2's perlbench changes the size of 894 functions (Amendment 1), so its twin difference is not zero. The rule
    decides, and its perlbench enters P_s as the rule says.
