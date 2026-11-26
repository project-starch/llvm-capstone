# -capstone-pool-base-at-use-dominator on nab, imagick and perlbench, alone and with LSRH (pre-registration, 2026-11-26)

The spec8 split (1430c473c2c6) found that -capstone-pool-base-at-entry costs nab +0.382 % and perlbench +0.369 %.
In perlbench, Perl_pp_modulo pays 5 instructions on each of 1.9 M calls, for constant loads on cold paths. The new
option (1a191ec0e8a9, default off) shares the base only for chains whose loads are all in loops. It places that base
at their common dominator outside every loop, behind a static frequency gate.

This run measures the option alone (UDOMk) and with LSRH (UDLS) on the split's three stages. The plan lists it as
optional work; it is its own pre-registration. Written before any boot of this run.

## Arms

The compiler is llvm-sinkland clang at 1a191ec0e8a9's source, sha256 35580b17947b74ec. Its version string names
0f948fd8340f, the same change before a comment-only amend. UDOMk's images equal those of the earlier build
(8eed5279c7b20f62) byte for byte on all three programs.

| arm | extra options | perlbench_r | imagick_r | nab_r |
|---|---|---|---|---|
| UDOMk | -capstone-pool-base-at-use-dominator | f311ffbdedd2c5f2 | 300f1ff9f400ae55 | 05c9125566a36943 |
| UDLS | that, plus -lsr-capability-exit-runtime-end -lsr-capability-exit-end-hoist | 04cadf36176f6996 | 0a8d3f90717d8666 | d145dabacb2b5329 |
| LSHOFF3 | LSHOFF's images, a third boot on this init | f3c8c80d71844564 | a666e331f10d73cc | da57e5473a33e6fd |

The reference is the split's LSHOFF boot, the same init and images.

With the option off, this compiler builds LSHOFF's images byte for byte (perlbench, imagick, nab and x264), and with
-capstone-pool-base-at-entry it builds POOL's.

## Static reading, before the boot

The comparison is by c4/spec8s/fnsig.py. A function whose code equals another arm's counts what that arm counted.

**UDOMk against LSHOFF:**
- perlbench changes 10 functions. Of those that run, only __capstone_cap_init, at 16,794 instructions, whose code
  equals POOL's (delta 0). Perl_pp_modulo, Perl_cast_uv, Perl_pp_int and Perl_sv_dec_nomg are LSHOFF's code.
- imagick changes 107 functions.
  - HorizontalFilter and VerticalFilter are POOL's: together with the other POOL-equal functions, -1,099,410.
  - Four executed functions have new code: WriteTGAImage (8,368,509 in LSHOFF), YShearImage, XShearImage and
    __capstone_cap_init.
- nab changes 9 functions.
  - ephi, eangl, md and getpdb are POOL's: -9,214,753.
  - mme34 (4,313,736,041 in LSHOFF) and makebonds have new code. mme34 is 1,813 instructions long, against 1,799 in
    LSHOFF and 1,815 in POOL.

**UDLS** combines the two options' versions:
- imagick: HorizontalFilter, VerticalFilter and NegateImageChannel are LSHON's (-2,775,592 together); WriteTGAImage
  is UDOMk's.
- nab: md and cap are LSHON's, ephi and eangl POOL's, and mme34 UDOMk's.
- perlbench: Perl_pp_modulo is LSHOFF's, and Perl_av_store LSRH's.

## Procedure

As the split: spec8s's make.sh, drive.py at shift 1, count8s.py and split8.py (SPLIT_ARMS names the arms), under the
QEMU lock. The boots run in the order LSHOFF3, UDOMk, UDLS, each taking and releasing the lock.

## Predictions

1. **Repeat: LSHOFF3 equals LSHOFF within ±100 per stage.**
2. **perlbench: UDOMk equals LSHOFF within ±100, and UDLS equals LSHOFF - 34 within ±100**, the -34 being LSRH's.
3. **imagick:**
   - UDOMk between -1,150,000 and -1,050,000 (-1,099,410 from POOL-equal code, plus the four new-code functions);
   - UDLS between -2,850,000 and -2,700,000 (-2,775,592 from LSHON-equal code, plus the same four).
4. **nab, which is not predicted in sign.**
   - UDOMk = -9,214,753 + Δ(mme34) + Δ(makebonds), and UDLS = UDOMk - 2,984,712 within ±1,000.
   - Under POOL, Δ(mme34) was +36,513,369. UDOMk is within the +0.05 % band (+3,577,406) only if Δ(mme34) is at most
     about +12.8 M.
   - The run reads Δ(mme34) per instruction.
5. **Functions unchanged against LSHOFF count the same, within ±1,000 per stage.**
6. **Outputs: every stage's hash set equals LSHOFF's.**

## What it decides

- If nab and perlbench are inside +0.05 % for UDOMk, the option gets an eight-stage twin gate like spec8-lsrh. UDLS
  then becomes the candidate that adds imagick's pool saving to LSRH's.
- If nab is outside the band, the option stays default off. The mme34 reading then says where its loop chains cost.
- Neither outcome touches the board.
