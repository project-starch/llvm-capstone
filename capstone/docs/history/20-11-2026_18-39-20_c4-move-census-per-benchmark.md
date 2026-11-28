# C4: register moves per benchmark, and the round trips among them

Programme item C4: "count the move round trips per benchmark, then fix". The earlier census (14 Nov, on gcc train01
and mcf only, branch compiler-memcpy-align-dispatch, history `14-11-2026_06-00-00_compiler-lever-census-c8-and-the-
memset-switch.md`) found "no concentrated idiom". This note counts every benchmark that has a per-instruction profile
on both sides.

Result:
- Strict round trips are negligible everywhere; the largest is xz at +0.024 % of riscv-base's instructions. No
  round-trip fix is warranted.
- The move excess that exists is concentrated. Its largest single site is one integer copy per iteration of lbm's
  stream-collide loop: 71.5 % of lbm's instruction gap, and a consequence of that loop's induction-variable shape
  (programme item 4).

## Instrument

- Classifier: `movctx.py` of the 14 Nov census, unchanged. Every executed `mv` and `movc` (`mv` on riscv-base) goes
  to one context: entry copy into a callee-saved register, argument setup, return value, return setup, round trip,
  or loop-carried/other. A round trip is `mv b, a` whose source was written by `mv a, b` within the previous eight
  instructions of the function, neither register written in between.
- Profiles: shift-1 (per-instruction) capmix, every command window, mapped onto each image by one load offset per
  image instance.
  - Capstone: arm FRCDKMSa3 for gcc (the paper's Full arm); arm FRCDKMS for the others, which have no FRCDKMSa3
    profile. Both arms use the same compiler, clang 29027126f39b, and the same flags.
  - riscv-base: the native RVAL profiles.
- Controls:
  - mcf's move excess reads +166.6 M, the 14 Nov reading (+166 M) on a different window and mapper.
  - The round-trip detector fires: `printf_core` reads the same count on both sides.

## Readings

Moves as Capstone minus riscv-base, in % of riscv-base's image instructions and as a share of the row's instruction
gap; strict round trips likewise:

| benchmark (input) | gap | move delta | share of gap | round-trip delta |
|---|---:|---:|---:|---:|
| gcc train01 | +12.66 % | +213.8 M, +1.97 % | 15.6 % | -0.47 M, -0.004 % |
| mcf test | -2.62 % | +166.6 M, +0.61 % | (gap negative) | +279 |
| lbm test | +1.21 % | +52.0 M, +0.86 % | 71.5 % | +645 |
| nab test | +14.01 % | +4.9 M, +0.08 % | 0.6 % | +5,126 |
| xz test | +0.32 % | +201.4 M, +0.41 % | 127 % | +12.06 M, +0.024 % |
| imagick test-ref | +1.90 % | +1.7 M, +1.78 % | 94 % | +178 |

Not counted:
- perlbench: the riscv-base side does not map (749 M executions land on non-instruction starts in the native image),
  unresolved.
- x264: no FRCDKMS-family profile.
- imagick: the test-ref profile covers only 95 M instructions of the convert step, so its row is not representative.

## Where the excess is

- **lbm.** All of it, 52,000,017, is one integer `mv s0, s1` per iteration of the loop in
  `LBM_performStreamCollideTRT`. The index is incremented into `s1` (`addi s1, s0, 0x14`), the exit test reads the
  old `s0` (`bgeu s0, t4`), and the copy closes the iteration. Two capability pointers beside it step by 0xa0. On
  riscv-base the same loop has one byte-offset index added to each base, and no copy. The copy goes when the
  induction-variable shape is changed: plan 2026-11-02 item 4, pointer induction variables in LBM.
- **xz.** `memcpy` +174 M `movc`; `rc_encode` +96 M and `sha_compress` +92 M `mv`.
- **mcf.** `spec_qsort` +161 M: `movc` replaces riscv-base's `ld`+`add` pairs, and the function as a whole runs fewer
  instructions (14 Nov).
- **gcc.** Spread: `bitmap_elt_ior`, `bitmap_set_bit`, `memset`, `walk_tree_1` and `memcpy`, 5 to 12 M each.

## Decision

Against the runtime lane's go/no-go (0.2 % of a benchmark's instructions), round trips do not reach it anywhere.
No round-trip change is built. The excess above 0.2 % is lbm's induction-variable copy, which moves to item 4, and
xz's `memcpy` capability moves, which are still to be explained.
