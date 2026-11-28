# -capstone-pool-base-at-use-dominator on nab, imagick and perlbench, alone and with LSRH: results (2026-11-26)

Pre-registration: PREREG.md, c49cd1e1fa4c, pushed before any boot.

**The option removes perlbench's cost but not nab's.**
- perlbench reads 0 under UDOMk: Perl_pp_modulo is LSHOFF's code again.
- nab reads +27,299,622 (+0.382 %), POOL's regression to within 1,006 instructions. All of it is in mme34, whose
  cost is the same as under -capstone-pool-base-at-entry, class by class.
- **So the option stays default off, and it gets no eight-stage gate.**
- imagick keeps POOL's saving (-1,100,234), and with LSRH (UDLS) LSHON's (-2,776,388).

Every prediction holds; prediction 4 states nab's value as a formula and gives no sign.

The compiler is llvm-sinkland clang sha256 35580b17947b74ec, the binary the pre-registration names.

## The boots

There were three boots under the QEMU lock, each taking and releasing it: LSHOFF3, UDOMk, then UDLS.
- They ran after the campaign twin and the LSRH repeat, as runtime's ordering required. The wrapper started them at
  19:55, and they ended at 21:13.
- Each boot took 166 to 172 s at shift 1. All three exited rc 0, with every stage rc 0 and none cut.
- The reference is the split's LSHOFF boot, on the same init and images.
- count8s.py now drops the init's tick lines from the output-hash block (tests/spec8-lsrh RESULTS, "A reader fix").
  The split's five arms read the same before and after that change.

## Per stage, against LSHOFF

| stage | LSHOFF | LSHOFF3 | UDOMk | UDLS |
|---|---:|---:|---:|---:|
| 544.nab_r | 7,154,811,027 | 0 | +27,299,622 (+0.382 %) | +24,314,902 (+0.340 %) |
| 538.imagick_r | 94,485,853 | -90 | -1,100,234 (-1.164 %) | -2,776,388 (-2.938 %) |
| 500.perlbench_r | 3,094,620,533 | 0 | 0 | -34 |

Every stage's outputs are equal across the four boots.

## Per function (split8.py)

| stage | arm | function | delta |
|---|---|---|---:|
| nab | UDOMk | mme34 | +36,514,371 |
| | | ephi | -9,230,424 |
| | | eangl | +16,032 |
| | | getpdb | -365 |
| | | md | +4 |
| | UDLS | mme34, ephi, eangl, getpdb | as UDOMk |
| | | md | -2,984,712 (LSRH's) |
| imagick | UDOMk | HorizontalFilter | -1,073,996 |
| | | VerticalFilter | -25,436 |
| | | YShearImage | -519 |
| | | XShearImage | -212 |
| | UDLS | HorizontalFilter | -2,432,236 |
| | | NegateImageChannel | -286,560 |
| | | VerticalFilter | -56,796 |
| perlbench | UDOMk | (none) | 0 |
| | UDLS | Perl_av_store | -32 |
| | | Perl_av_clear | -2 |

## Where mme34's cost is

mme34, by capmix3 instruction class, over the same windows and address mapping as split8 (c4/spec8s/mme-class.py).
The totals equal split8's, 4,313,736,041 and 4,350,250,412.

| class | UDOMk - LSHOFF | POOL - LSHOFF |
|---|---:|---:|
| caparith | +22,921,755 | +22,922,757 |
| alu | +15,286,512 | +15,286,512 |
| load | +7,517,004 | +7,517,004 |
| fp | +751,500 | +751,500 |
| capload | -9,959,391 | -9,961,395 |
| store | -1,002 | -1,002 |
| capstore | -2,007 | -2,007 |
| branch | 0 | 0 |

**The two options give mme34 the same dynamic cost, class by class.**
- They differ by 2,004 capability loads and 1,002 capability-arithmetic instructions out of 36.5 M.
- Branches are unchanged, so the loop trip counts are the same.
- The +18,194 instructions per call (2,007 calls) are in mme34's loops, not in building the base: building it is a
  pair per call.

**What this rules out, and what it leaves.**
- Under the use-dominator rule, the shared base is built at the common dominator of in-loop chains, outside every
  loop. Under the entry rule, it is built at entry. mme34 costs the same either way, so for this function the
  base's placement is not what costs.
- What both options do is share one base across mme34's in-loop constant-pool chains. That is the candidate.
- **No mechanism is claimed:** mme34's loops under the two codes were not read instruction by instruction. The
  pre-registration's "per instruction" reading is given here per class.

## Predictions

1. **Repeat, LSHOFF3 = LSHOFF within ±100 per stage: HOLDS.** nab 0, imagick -90, perlbench 0.
   - The -90 is the libc offset, in the same four functions as in the split (__fwritex, memset, printf_core,
     memcpy).
   - It remains UNEXPLAINED: what selects the value has not been read.
2. **perlbench: HOLDS.** UDOMk 0, and UDLS -34, which is LSRH's -34.
3. **imagick: HOLDS.**
   - UDOMk -1,100,234, in [-1,150,000, -1,050,000];
   - UDLS -2,776,388, in [-2,850,000, -2,700,000].
4. **nab: HOLDS as stated.**
   - The POOL-equal functions (ephi, eangl, md, getpdb) sum to -9,214,753 exactly. Δ(mme34) is +36,514,371, and
     Δ(makebonds) is 0. Their sum is the changed-function total, +27,299,618, and UDOMk's +27,299,622 is that plus
     alloc_slot's +4.
   - UDLS - UDOMk = -2,984,720, against -2,984,712 ± 1,000.
   - **UDOMk is outside the +0.05 % band (+3,577,406).** Δ(mme34) is +36.5 M, above the +12.8 M the band allowed.
5. **Functions unchanged against LSHOFF count the same, within ±1,000 per stage: HOLDS.**
   - nab +4 (alloc_slot);
   - imagick -93 (UDOMk) and -90 (UDLS), the libc offset;
   - perlbench 0.
6. **Outputs: HOLDS.** Every stage's hash set equals LSHOFF's.

## What it decides

- Per the pre-registration, nab is outside the band, so **-capstone-pool-base-at-use-dominator stays default off.**
  Neither it nor UDLS gets an eight-stage gate.
- The pool saving on imagick (-1.10 M) stays unavailable without nab's cost.
- LSRH alone remains the candidate. Its board form is pre-registered separately (tests/spec8-lsrh-board).
- A pool-base variant that leaves mme34's loops unshared would need its own pre-registration. The class reading
  above is where it would start.

## What this does not show

- mme34's mechanism, below the class level.
- Cycles, since these are instruction counts on the trap twin.
- P1.
