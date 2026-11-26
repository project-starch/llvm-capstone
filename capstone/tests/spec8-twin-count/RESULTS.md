# B5TL1's eight SPEC test stages on the twin, LSHOFF against LSHON: results (2026-11-26)

Pre-registration: PREREG.md, f9745ba5ae33, with amendment 1 (cfbbbe025981), both pushed before either boot started.

**Prediction 3 fails, so the board form's condition does not hold.** nab_r is +0.340 % and perlbench_r +0.369 % under
LSHON, against a band of +0.05 %. The other six stages are within it.

What holds:
- **imagick, both variants, inside their bands:** LSHON saves 2,775,976 and lsrh 1,676,157;
- **outputs:** every stage's outputs are equal in both arms.

The reader's control (prediction 1) misses its band by 2,120 instructions. All of them are in command-line and
environment string code.

## The boots

Two boots under the QEMU lock, platform as pre-registered:

| arm | ran | duration | result |
|---|---|---|---|
| LSHOFF | 04:13 to 06:29 | 8,169 s | 8/8 stages rc 0 |
| LSHON | 06:32 to 08:48 | 8,140 s | 9/9 stages rc 0 (with imagick-lsrh) |

No stage was cut. Every native image's hash is the one PREREG.md lists (spec8 images.sha against the arms' .strip.dom).

The reader is count8.py:
- it reads the per-command SNAP pairs (mark1, mark2), and every pair reads marks 1,2;
- one sum per native command, over the image range from its "enters" base to the ELF's loaded span.

The kernel prints the process name cut to 15 characters (`lbm_r_base.nati`). The reader's first filter
expected the full name and would have matched nothing. It was fixed before any LSHON row was read, and it now pairs 19
(LSHOFF) and 20 (LSHON) native commands with as many enters lines.

## Per stage

| stage | LSHOFF | LSHON | diff | % | outputs |
|---|---:|---:|---:|---:|---|
| 519.lbm_r | 6,092,401,820 | 6,092,401,820 | 0 | +0.000 | equal |
| 505.mcf_r | 26,644,261,994 | 26,644,603,084 | +341,090 | +0.001 | equal |
| 544.nab_r | 7,154,811,027 | 7,179,124,927 | +24,313,900 | **+0.340** | equal |
| 557.xz_r (12 commands) | 49,619,823,268 | 49,619,823,292 | +24 | +0.000 | equal |
| 502.gcc_r | 21,710,028 | 21,710,735 | +707 | +0.003 | equal |
| 538.imagick_r | 94,485,673 | 91,709,697 | -2,775,976 | -2.938 | equal |
| 538.imagick_r-lsrh | 94,485,673 | 92,809,516 | -1,676,157 | -1.774 | equal |
| 525.x264_r | 506,620,171,651 | 506,640,767,224 | +20,595,573 | +0.004 | equal |
| 500.perlbench_r | 3,094,620,533 | 3,106,027,546 | +11,407,013 | **+0.369** | equal |

nab's and perlbench's increases are also in the stage's all-user count, which does not depend on the image range:
- nab: +23,826,000 (7,170,708,946 to 7,194,534,791);
- perlbench: +11,400,000 (3,104,281,341 to 3,115,681,818).

gcc's stage is one compile of t1.c, which takes 1.9 s. Its only diagnostic is a printf-conflict warning, and its
.s has the same hash in both arms and in earlier boots of these stages.

## Predictions

1. **Control, LSHOFF imagick = 94,483,553 ± 1,000: FAILS.** It read 94,485,673, +2,120. The two boots were compared
   function by function, with both tables bucketed to 16 bytes (spec8 ctl8.py), and every one of the 2,120 is in
   string and path code fed by the command line and environment:
   - strncmp +440, strlen +360, strchrnul +328, getenv +280;
   - CopyMagickString +190, ParseCommandOption +185, GetPathComponent +178;
   - nothing else moves by more than 62.

   No filter function differs. Each boot's total also matches its own reading exactly.

   The band came from a 44-instruction move when only the program name changed. This boot also changes the
   environment and the working directory: the stage's init in place of a single-program init. The reader is not at
   fault. The miss is a cross-boot offset of about 2 × 10^3, which cancels in the LSHOFF-LSHON rows, because both arms
   use the same init.
2. **Outputs, every stage's hash set equal: HOLDS.** All nine stages are equal, lsrh included, and imagick's
   test_output.tga reads 3f70992184b213de in all three runs. The sets range from 5 files (lbm) to 561 (perlbench).
3. **No stage worse than +0.05 %: FAILS.** nab_r is +0.340 % and perlbench_r +0.369 %. mcf (+0.001 %), gcc
   (+0.003 %) and x264 (+0.004 %) are inside the band, and lbm and xz are flat.
4. **imagick, LSHON: a reduction between 2.55 M and 2.85 M: HOLDS**, at 2,775,976 (2.94 %).
5. **imagick, lsrh: a reduction between 1.50 M and 1.75 M: HOLDS**, at 1,676,157 (1.77 %).

   Per function, LSHON against LSHOFF, in 16-byte buckets:
   - HorizontalFilter -2,432,243;
   - NegateImageChannel -286,558, against the -286,560 predicted for the LSR part;
   - VerticalFilter -56,793.

   lsrh was not split per function.

## What the readings do not show

- **Which option, or which function, costs nab and perlbench.** LSHON carries three options at once.
  - A per-function split at shift 4 is not reliable at function boundaries. A 16-byte bucket is credited to the first
    instruction that starts in it, so when the layout shifts, a function's hot entry moves into its neighbour's
    bucket.
  - The same imagick table shows the effect at small scale: tolower +65,564 against isxdigit_l -65,564.
  - In perlbench, the large opposite-signed pairs are neighbours in the address space:
    - S_pushav -15.2 M against Perl_pp_padsv +15.2 M, inside a 0x1200-byte cluster that also holds a
      __capstone_cap_init, Perl_pp_nextstate, Perl_pp_gv and Perl_pp_and;
    - the Perl_cast_* triple.
  - In nab, log -5,124,002 against hypot +5,124,002 are adjacent. LSHON's libm block sits 0xac bytes later, which is
    not a multiple of 16.
  - The stage totals sum the whole image range and are unaffected.
  - One candidate is away from a boundary: nab's mme34, +36.5 M. It starts at 0x144a8 in both arms and is 76 bytes
    longer in LSHON (its successor ephi moves from 0x1698c to 0x169d8). That is a lead, not an attribution.
  - Finding the cause needs shift-1 profiles and arms that split the options. Neither was run here.
- Cycles. These are instruction counts on the trap twin.
- P1.
