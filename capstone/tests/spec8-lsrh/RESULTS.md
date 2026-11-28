# LSRH on B5TL1's eight SPEC test stages, with an LSHOFF repeat: results (2026-11-26)

Pre-registration: PREREG.md, 8bf6a86af81f, pushed before either boot.

**The gate holds on all eight stages, and every prediction holds.** LSRH (-lsr-capability-exit-runtime-end
-lsr-capability-exit-end-hoist) costs at most +0.004 % on any stage (x264), and it keeps:
- imagick's -1,676,067 (-1.774 %);
- nab's -2,984,712 (-0.042 %).

The compiler is llvm-sinkland clang, sha256 b95bc7ce930075de, for every arm.

## The boots

There were two boots under the QEMU lock, each taking and releasing it:
- **LSRH:** 09:59:45 to 12:19:16, 8,370 s.
- **LSHOFFR:** 14:00:59 to 16:21:32, 8,433 s.
  - Its first queueing was withdrawn before it booted, so as not to hold the lock between two boots of another
    lane's twin chain.
  - It was requeued behind that chain's end marker.

Both boots exited rc 0, with all eight stages rc 0 and none cut. LSHOFF is spec8's boot (8abd11010c5b), not re-run.

## Per stage

| stage | LSHOFF | LSRH - LSHOFF | LSHOFFR - LSHOFF (r) | band: 0.05 % + r | LSRH within |
|---|---:|---:|---:|---:|---|
| 519.lbm_r | 6,092,401,820 | 0 | 0 | 3,046,201 | yes |
| 505.mcf_r | 26,644,261,994 | +341,090 (+0.001 %) | 0 | 13,322,131 | yes |
| 544.nab_r | 7,154,811,027 | -2,984,712 (-0.042 %) | 0 | 3,577,406 | yes |
| 557.xz_r | 49,619,823,268 | +24 | 0 | 24,809,912 | yes |
| 502.gcc_r | 21,710,028 | +696 (+0.003 %) | 0 | 10,855 | yes |
| 538.imagick_r | 94,485,673 | -1,676,067 (-1.774 %) | +268 | 47,511 | yes |
| 525.x264_r | 506,620,171,651 | +20,594,314 (+0.004 %) | 0 | 253,310,086 | yes |
| 500.perlbench_r | 3,094,620,533 | -34 | 0 | 1,547,310 | yes |

The largest use of a band is x264's, 8.1 % of it. gcc's is 6.4 %.

## Predictions

1. **Repeat, LSHOFFR = LSHOFF within ±1,000 on seven stages, and x264 within ±0.001 %: HOLDS.**
   - Seven stages read 0, x264 among them. So x264's first repeat reads 0 instructions on 506.6 G.
   - imagick reads +268.
   - imagick's count is now known to take at least three values for the same image:
     - 94,485,673 in spec8's LSHOFF boot and 94,485,941 in LSHOFFR, on this init;
     - 94,485,853 in the split's LSHOFF boot, on the split's init.
   - LSRH's imagick, 92,809,606, is 90 above spec8's boot of the same image. This is the libc offset that the split
     recorded as UNEXPLAINED (tests/spec8-split, prediction 1), and it remains UNEXPLAINED; no mechanism is claimed.
2. **Identity rows: HOLDS.**
   - lbm 0.
   - mcf 26,644,603,084 and xz 49,619,823,292, each equal to spec8's LSHON reading to the instruction.
3. **nab -2,984,712 and perlbench -34: HOLDS**, both exactly the split's readings.
4. **imagick -1,676,157 ± 1,000: HOLDS** at -1,676,067. The 90 is the offset in prediction 1.
5. **x264 between 0 and +0.010 %: HOLDS** at +20,594,314 (+0.004 %).
   - spec8's LSHON read +20,595,573 on its image. Since LSHON's x264 image is not LSRH's, the two are not expected to
     be equal.
6. **gcc within ±0.05 %: HOLDS** at +696.
7. **Outputs equal: HOLDS.** Each stage's output-hash set in LSRH and in LSHOFFR equals LSHOFF's (561 files for
   perlbench).
8. **The gate: HOLDS on every stage.**

## A reader fix, made after the repeat was read

count8.py first printed perlbench's outputs as DIFFER between LSHOFF and LSHOFFR.
- The whole difference was one console line, `TRAPTEST: tick 8424.64`, the init's periodic tick. It landed inside
  the @@SPEC-HASH block, so the set had 562 entries instead of 561.
- Every file hash was in both sets.
- The reader now drops `TRAPTEST: tick ` lines from the hash block.
- The negative control was a copy of LSHOFFR's log with one perlbench output hash altered by one digit. It still
  reads DIFFER, and every other stage reads EQUAL.
- Removing lines can only turn a difference made of tick lines into EQUAL. No other spec8 reading changes: LSHOFF
  against LSRH was EQUAL before and after the fix.

## What this decides, and what it does not

- **The board form may return,** under a pre-registration of its own:
  - an in-session bracket LSHOFF, LSRH, LSHOFF2;
  - its imagick band written from -1,676,067.
- Cycles: these are instruction counts on the trap twin.
- P1, where the run-time endpoint does not hold.
