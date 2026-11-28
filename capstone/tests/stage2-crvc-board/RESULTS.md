# Stage 2 on Line B, the CRVC pair on its seal: results (2026-11-28)

Pre-registration: PREREG.md, 4d0f6b69116f, with Amendments 1-7. Each was pushed before the boot it governs:
- f27d0ba8979f: the seal, the kernel K, the firmware and the twin checks;
- ade77cfc5ca3: the firmware's test order and print;
- b83fd20bfa39: the void KTEST runs;
- 232c5755314d: the twin readings;
- 92aef6f99a7b: the images and the boot line;
- a09866296ebb: K0's predicate;
- e8480580138b: the seal control's spread.

**VERDICT: GO, carried by gcc. The stage-2 seal is proposed to the lead.** The seal control is not complete: it is
UNDECIDED on perlbench and nab, so the condition for replacing the campaign seal is not met.
- **gcc t1.c** reads 0.090 s (1.62 %) faster with CRVC.
  - Every ON boot lies at least eight ticks (eight to ten) below every OFF boot; the means differ by nine.
  - Placement does not move gcc (P_gcc = 0.00 s), and 1 % of OFF is 0.056 s, so the gain counts in full.
  - That is r = 0.179 of its 9.068 % fetch saving, inside the band [0, 0.37].
- **perlbench makerand** does not move: +0.29 s (+0.10 %). Its 17.3 % fetch saving reaches no measurable time, against
  a placement spread of 11.38 s.
- **imagick and nab** are flat. nab, the same binary in both images, holds both ways.
- No stage reads worse. The drift control holds. Outputs are equal in all seven boots.
- **The firmware control** read as predicted on every boot: one redirect and SIGBUS on N, the refusal on b42c8c644.
- **Predictions:** "GO, carried by perlbench" is refuted on its carrier, and "perlbench reads a saving" is refuted.
  K0's own window is refuted on perlbench and nab (see "The seal control").

## The session

k-b2-20261127T0408 on board 2, run by the board lane (orchestrator orch-k.sh b27f8fbd751fa995) in the boot line's
order.
- **Its gates passed at 17:24:37Z:**
  - the previous session's .done;
  - this lane's placement RESULTS, present at c89f177d0403;
  - the boot line verbatim;
  - both images' full sha256 and size;
  - the stage roots, preflight and graders.
- **Every boot read on its first attempt,** with no rerun and no stop. The bitstream was read back before and after
  each boot.
- **Graders:** the four timed stages were graded by v2 and v1 on a copy with only the K-crvcmis QSTEP block cut out
  (the board lane's k-strip, 239 B on K0 and 1,587 B on N). Every stage of every boot passes. The crvcmis predicate
  was checked on the full capture by the board lane's checker and by this lane's reader. Both pass on the twin captures
  in their own forms and miss on the crossed pairs.
- **This lane read every time below from the captures,** after checking each sha256 against the board lane's row.

| boot | image | bitstream | uart sha256 | RUN END |
|---|---|---|---|---|
| K0 | S-LSHOFFL-K (ac582c15…) | b42c8c644 | b63bb0adaca1c0e6 | rc=0 17:52:19Z |
| FLASH-N | | 796af23d9: resident after the power cycle, readback = the seal's name | | rc=0 17:54:41Z |
| OFF1 | S-LSHOFFL-K | N | eb8efb9ec74d2e2e | rc=0, the size proof on N: 25,856,520 B reached /init |
| ON1 | S-CRVCBN-K (a3b08c74…) | N | 9e2918f7cdc01407 | rc=0 18:50:14Z |
| OFF2 | S-LSHOFFL-K | N | bfa34d8379be447d | rc=0 19:18:01Z |
| ON2 | S-CRVCBN-K | N | a8a5dc9c2ddddb41 | rc=0 19:45:44Z |
| OFF3 | S-LSHOFFL-K | N | 3e96579d22409a27 | rc=0 20:13:29Z |
| ON3 | S-CRVCBN-K | N | 7b3fdf3b783ab330 | rc=0 20:41:17Z |
| FLASH-B42 | | b42c8c644: resident after the power cycle, readback b42c8c644 | | rc=0 20:43:38Z |

**Outputs:** this lane compared each boot's four hash blocks with K0's, sorted, with tick lines removed (5, 8, 557 and
5 lines). All six N boots are equal to K0, and K0 is equal to the placement session's LSHOFFL. Altering one line makes
the comparison fire. No capture has a "sublet: fault" line.

**The slot:** the board queue moved twice. The session ran after camp8x-b2-20261127T0342's .done, the x264b session
Amendment 5's slot clause names under its earlier run name.

## The pair, on N

t is up1 - up0 on each stage's @@SPEC-END run line, in seconds. spread_s = max(P_s, range(OFF)), with P_s from
tests/stage2-placement-b42 RESULTS (c89f177d0403).

| stage | OFF1, OFF2, OFF3 | ON1, ON2, ON3 | Δ = mean(ON) - mean(OFF) | spread | gain | no worse | r |
|---|---|---|---:|---:|---|---|---:|
| gcc | 5.56, 5.56, 5.55 | 5.47, 5.46, 5.47 | **-0.090 (-1.62 %)** | 0.01 | **YES** | holds | 0.179 |
| imagick | 11.25, 11.24, 11.24 | 11.22, 11.25, 11.25 | -0.003 (-0.03 %) | 0.05 | no | holds | |
| perlbench | 276.18, 277.35, 277.10 | 277.03, 277.31, 277.15 | +0.287 (+0.10 %) | 11.38 | no | holds | -0.006 |
| nab | 728.39, 728.23, 728.31 | 728.19, 728.21, 728.08 | -0.150 (-0.02 %) | 2.21 | no | holds | |

- **gcc's gain:** max(ON) = 5.47 < min(OFF) - 0.02 = 5.53, and -Δ - P_gcc = 0.090 ≥ 0.01 × 5.557 = 0.056.
- **No stage is worse:** each Δ is within 0.0005 × mean(OFF) + spread + 0.02.
- **nab both ways:** |Δ| = 0.150 against 0.0005 × 728.31 + range 0.16 + 0.02 = 0.544. It holds, so the images differ in
  nothing nab sees.
- **The drift control** (imagick val): OFF 5.43, 5.43, 5.42; ON 5.42, 5.42, 5.43. |Δmean| is 0.003 against 0.030.
- **The firmware's control, on every N boot:**
  - exactly one `FWCRVC redirect=1 emulate-u=0`, spliced into the program's "loading" line, as predicted;
  - `@@CRVCMIS rc=135`;
  - "crvcmis.dom[...] binds in the CRVC mode";
  - no "no trap";
  - the banner before "Linux version";
  - no FWCRVC line inside a timed stage.
- **The SIGBUS pc:** the kernel reports "unhandled signal 7 ... at 0x3ff7f02ac6 in crvcmis.dom[3ff7f00000+d000]". That
  is ELF 0x12ac6, the 16-bit word 0x210c, the c.ldc itself. So the redirected trap reached S with epc at the c.ldc,
  as rtl predicted for mepc.
- **emulate-u = 0 in every boot:** no misaligned access from U was emulated before the control in any N boot, OFF or
  ON.

## The seal control

seal_s = mean(OFF on N) - t(K0). Under Amendment 7 its allowance is 0.0005 × t(K0) + max(range(OFF_s), B_s) + 0.02 s.
K0 must first lie within the five stock S-LSHOFFL boots on b42c8c644 (the ZCB pair's OFF1-OFF3, placement's LSHOFFL
and LSHOFFL2), ± 0.0005 × max + 0.02.

| stage | K0 | the stock window | K0 | seal_s | allowance | seal control |
|---|---:|---|---|---:|---:|---|
| gcc | 5.56 | [5.53, 5.58] | holds | -0.003 | 0.033 | **holds** |
| imagick | 11.25 | [11.19, 11.27] | holds | -0.007 | 0.046 | **holds** |
| perlbench | 273.57 | [277.42, 279.32] | **FAILS, 3.85 s below** | (+3.31) | | **UNDECIDED** |
| nab | 728.65 | [727.09, 728.33] | **FAILS, 0.32 s above** | (-0.34) | | **UNDECIDED** |

- **K0's crvcmis read as predicted:**
  - "CRVC image refused: this hart has no CRVC mode";
  - the syntax-error line;
  - `@@CRVCMIS rc=2`;
  - no redirect line;
  - the banner before Linux.
- **K0's two failures say that K, its firmware or the K overlay moves perlbench and nab on b42c8c644.**
  - K0 runs the stock LSHOFFL natives; only the kernel, the firmware and the overlay's added stage differ.
  - Which of them carries it is not read. Placement alone moved perlbench by 11.38 s, so a 3.85 s move from a
    different kernel or initramfs layout fits the same scale.
  - **Under Amendment 5:** the seal's own cost on perlbench and nab is UNDECIDED. The parenthesised seal_s values are
    not readings.

## Predictions

1. **perlbench reads a saving, r_perl in (0, 0.37]: REFUTED.** It read Δ = +0.287 s, r ≈ 0.
2. **gcc reads flat or a saving (Δ ≤ 0), no more than 0.19 s: HOLDS.** It read -0.090 s. Whether it would clear
   r ≥ 0.150 was not predicted; it read r = 0.179.
3. **imagick flat, |Δ| < 0.03 s: HOLDS,** at -0.003 s.
4. **nab flat under rule 3: HOLDS,** at |Δ| = 0.150 ≤ 0.544.
5. **The seal control holds on every stage:**
   - **HOLDS** on gcc and imagick;
   - **UNDECIDED** on perlbench and nab, because K0's own window failed there (Amendment 5). It is neither held nor
     refuted.
6. **"The predicted verdict is GO, carried by perlbench": the verdict, GO, HOLDS; its carrier is REFUTED.** gcc
   carries it, and perlbench reads nothing.
7. **K0 holds on all four stages** (Amendment 5's own prediction): **REFUTED** on perlbench and nab.

**For the band:**
- gcc's r = 0.179 lies inside [0, 0.37], between P1's two calibrations (0.37 perl, 0.05 xz).
- perlbench's r ≈ 0 sits at the band's floor. The band rejects only r > 0.37 or a slowdown beyond the spread, and
  neither occurred.

## What it decides

- **By "What decides": GO.** gcc carries the gain beyond placement, no stage is worse, nab holds both ways, and the
  drift control holds. Stage 2 is proposed to the lead.
- **What the UNDECIDED hides:** if K0 is representative of S-LSHOFFL-K on b42c8c644, seal_perl is +3.31 s against
  an allowance of 1.74 s. That is, the seal would cost perlbench about 1.2 %, more than gcc's gain is worth in
  seconds. One K0 boot cannot say whether it is, so this is a conditional, not a reading.
- **Replacing the campaign seal is not met.** That requires GO, the seal control holding, and the lead's agreement.
  The seal control is UNDECIDED on perlbench and nab, so N does not replace b42c8c644. Board 2 is back on b42c8c644,
  read back.
- **The size of what is proposed:**
  - gcc 1.6 %;
  - perlbench, the campaign's heaviest stage, nothing measurable;
  - imagick and nab nothing.
  - What it would cost the campaign's other programs is the seal control's question, and on perlbench and nab that
    is not answered.

## What this does not show

- **Why perlbench's 17.3 % fetch saving reaches no time.** No I$ or front-end counter is read, and perlbench's
  placement spread (11.38 s) is about forty times the Δ.
- **Which of K, the firmware and the K overlay moved K0's perlbench and nab.** A K0-against-stock pair on b42c8c644
  that varies one of the three would read it.
- x264, mcf, xz and lbm. P1. Train inputs.

## Addendum (2026-11-28): Amendment 8's reading, the seal control from a three-boot K reference

Amendment 8 (bf8c9c6492b4) was pushed after these RESULTS and before either boot. It added two boots of S-LSHOFFL-K on
b42c8c644, back to back, with no flash: k0b-b2-20261127T2052, run by the board lane.
- Both boots read on their first attempt, and the readback named b42c8c644 before and after each.
- v2 and v1 pass all four stages on each. Outputs are equal to the placement LSHOFFL's, with 575 hashes per boot.
- crvcmis held K0's predicate on both: the refusal, `@@CRVCMIS rc=2`, no redirect line.
- This lane read each capture after checking its sha256: K0b e1358df999010fed, K0c 570e7d2166eb0151.

| stage | K0, K0b, K0c | mean(K) | mean(OFF on N) | seal_s | allowance | seal control |
|---|---|---:|---:|---:|---:|---|
| gcc | 5.56, 5.57, 5.57 | 5.567 | 5.557 | -0.010 | 0.033 | **holds** |
| imagick | 11.25, 11.26, 11.23 | 11.247 | 11.243 | -0.003 | 0.056 | **holds** |
| perlbench | 273.57, 276.61, 276.67 | 275.617 | 276.877 | +1.260 | 3.258 | **holds** |
| nab | 728.65, 728.37, 728.04 | 728.353 | 728.310 | -0.043 | 0.994 | **holds** |

The allowance is 0.0005 × mean(K) + max(range(OFF), range(K), B_s) + 0.02 s.

- **Amendment 8's prediction is REFUTED.** It predicted that the repeats would land near K0's 273.57 s and that the
  seal would cost perlbench time.
  - The repeats read 276.61 and 276.67, so K0's perlbench was one boot's excursion, 3 s below its own repeats.
  - **rtl's prediction holds,** written before the boots: seal_s = 0 on programs that do not set the mode, since with
    crvc_q = 0 the change adds no stall and no flop on any path.
- **The seal control now holds on all four stages.** RESULTS' UNDECIDED stands as what one K0 boot could read. This
  reading adds the cost, and it regrades nothing.
  - **perlbench's "holds" is weak:** no seal cost is detected, which is not the same as no cost. It read +1.26 s
    (+0.46 %) against an allowance of 3.26 s, and that allowance is almost all range(K) = 3.10 s, which K0's own
    excursion set. At this noise the test cannot tell 1.26 s from zero. Without K0, mean(K0b, K0c) = 276.64 gives
    +0.24 s.
  - On gcc, imagick and nab the allowances are small (0.03-0.99 s) and "holds" means what it says.
- **K0's own window against the stock boots stays refuted on perlbench and nab.** That refutation was K0's single
  reading. K0b and K0c (276.61, 276.67) also lie below the stock window's floor of 277.42. Whether K, its firmware or
  the overlay moves perlbench on b42c8c644 by about 1 s is not decided here.
- **The condition for replacing the campaign seal** (GO, the seal control holding, and the lead's agreement) now
  waits only on the lead.
  - What is offered: gcc t1.c 1.6 % faster with CRVC. perlbench, imagick and nab do not move. On programs without the
    mode, no seal cost is detected: on gcc, imagick and nab within small allowances, and on perlbench at most about
    +0.46 %, which this reading cannot distinguish from zero.
  - Board 2 is on b42c8c644, read back.
