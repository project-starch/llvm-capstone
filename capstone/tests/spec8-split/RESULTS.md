# Which option costs nab and perlbench: results (2026-11-26)

Pre-registration: PREREG.md, 08345389bc98, pushed before the first boot.

**-capstone-pool-base-at-entry carries both of spec8's regressions: nab_r +27,298,616 (+0.382 %) and perlbench_r
+11,407,047 (+0.369 %).** lsr-runtime-end plus the hoist (LSRH) costs neither stage:
- nab_r -2,984,712 (-0.042 %), all of it in md;
- perlbench_r -34;
- imagick_r still -1,676,247 (-1.774 %).

Every prediction holds.

## The boots

Five boots under the QEMU lock, each taking and releasing it, between 09:10 and 09:41. They interleaved with
another lane's queue.
- Every boot exited rc 0, with all three stages rc 0 and none cut.
- Each took 164 to 183 s at shift 1.
- Every native image's hash is the one PREREG.md lists.

The reader has two parts:
- count8s.py, spec8's count8.py at shift 1;
- split8.py, which reads the same SNAP pairs per function, credits each instruction to the function whose llvm-nm -S
  range holds it, and joins that with fnsig.py's changed/unchanged classification. No counted instruction fell outside
  a function symbol.

split8.py had two errors, both fixed before any arm other than LSHOFF was read.
- It stored an empty entry for every SNAP outside the wanted pair, and so refused to read.
- It did not stop at the FINAL table, which follows perlbench's mark2 table because perlbench is the last command
  here. That added the at-exit cumulative counts and doubled perlbench, to 6.28 G.

After both fixes, both readers agree on all three LSHOFF stages.

## Per stage, against LSHOFF

| stage | LSHOFF | POOL | LSRH | LSHON | LSHOFF2 |
|---|---:|---:|---:|---:|---:|
| 544.nab_r | 7,154,811,027 | +27,298,616 (+0.382 %) | -2,984,712 (-0.042 %) | +24,313,904 (+0.340 %) | +4 |
| 538.imagick_r | 94,485,853 | -1,099,819 (-1.164 %) | -1,676,247 (-1.774 %) | -2,775,976 (-2.938 %) | -90 |
| 500.perlbench_r | 3,094,620,533 | +11,407,047 (+0.369 %) | -34 | +11,407,013 (+0.369 %) | 0 |

Every stage's outputs are equal across the five boots.

nab and perlbench count exactly what they did in spec8's LSHOFF boot, under a different init: 7,154,811,027 and
3,094,620,533. imagick counts 94,485,853 here against 94,485,673 there, because its command-line code reads the
environment.

## Per function

| stage | arm | function | delta |
|---|---|---|---:|
| nab_r | POOL | mme34 | +36,513,369 |
| | | ephi | -9,230,424 |
| | | eangl | +16,032 |
| | | others | -361 |
| | LSRH | md | -2,984,712 |
| | LSHON | mme34, ephi, md, eangl, getpdb | each exactly as in POOL or LSRH |
| perlbench_r | POOL | Perl_pp_modulo | +9,501,120 |
| | | Perl_cast_uv | +1,900,224 |
| | | Perl_sv_dec_nomg | +5,688 |
| | | Perl_scan_num | +15 |
| | LSRH | Perl_av_store | -32 |
| | | Perl_av_clear | -2 |
| imagick_r | POOL | HorizontalFilter | -1,073,996 |
| | | VerticalFilter | -25,436 |
| | LSRH | HorizontalFilter | -1,358,240 |
| | | NegateImageChannel | -286,560 |
| | | VerticalFilter | -31,360 |

**The mechanism in perlbench: Perl_pp_modulo runs 1,900,224 times, at 101 instructions per call in LSHOFF and 106 in
POOL.** All five extra instructions are on every call's path:
- three (auipc, addi, scc s7, gp) build a pool base that this path never uses;
- two save and restore the callee-saved register (s9) that holding it costs.

That is the option's design: the base is built once at entry rather than at its uses. In a short function with a
cold use, it costs on every call.

mme34 runs 2,007 times, at 18,193 more instructions per call under POOL. Its code differs throughout, in register
allocation. Where in its loops the 18,193 fall was not read.

## Predictions

1. **Repeat, LSHOFF2 = LSHOFF ± 100 per stage: HOLDS.** nab +4, imagick -90, perlbench 0.
   - The -90 is all in libc: __fwritex -38, memset -23, printf_core -15, memcpy -14.
   - The four functions take one of two values from boot to boot. LSRH and LSHOFF2 show -90 against LSHOFF in the
     same four. POOL and LSHON show 0, like LSHOFF. What selects the value is not shown.
   - nab's +4 is alloc_slot, in the runtime's allocator.
2. **Reproduction: HOLDS.** LSHON against LSHOFF:
   - nab +24,313,904, against spec8's +24,313,900;
   - perlbench +11,407,013 and imagick -2,775,976, both equal to spec8's.
3. **Unchanged functions count the same, within ±1,000 per stage: HOLDS.** The largest sums are -90 (imagick
   libc) and +4 (nab alloc_slot). Every other arm and stage reads 0.
4. **perlbench: HOLDS.**
   - LSRH -34;
   - POOL +11,407,047 against LSHON's +11,407,013, a difference of 34, which is LSRH's -34;
   - every instruction of POOL's delta is in POOL-changed functions.
5. **nab: HOLDS.**
   - POOL +27,298,616, at least +20 M;
   - LSRH -2,984,712, within ±5 M;
   - POOL plus LSRH is +24,313,904, LSHON's reading exactly;
   - mme34's delta is +36,513,369 in both POOL and LSHON.
6. **imagick controls: HOLDS.**
   - POOL -1,099,819, against -1,099,810 ± 1,000;
   - LSRH -1,676,247, against -1,676,157 ± 1,000; the 90 is the libc offset in prediction 1.
7. **Outputs equal across the five boots: HOLDS.**

## What this does not show

- **LSRH on the other five stages.**
  - By whole-image identity, LSRH's lbm is LSHOFF's, and its mcf and xz are LSHON's (spec8: +0.001 %, +0.000 %).
  - Its gcc and x264 images are unread.
  - An eight-stage LSRH boot, with an LSHOFF repeat on the eight-stage init, is pre-registered separately.
- **x264's cross-boot repeatability.** x264 has the largest stage, and these boots do not include it.
- Cycles. These are instruction counts on the trap twin.
- P1.
