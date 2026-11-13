# C8: inline 16-byte copies of unknown alignment with a run-time alignment test

**What it changes.** C-81 sends a constant-size copy of a granule or more to a `memcpy` call when the compiler cannot
prove 16-byte alignment, because an `align 8` copy can still hold a capability at a 16-aligned address. The C-81
comment in `findOptimalMemOpLowering` says what was missing: a run-time alignment test in the expansion.
`-capstone-memcpy-align-dispatch` (default off; `CapstoneMemcpyAlignDispatch.cpp`, the last IR pass before selection)
adds that test for the one size where a two-way test matches the runtime's memcpy:

- both cursors 16-aligned: one `ldc`/`stc`, and the tag survives;
- otherwise: two `ld`, then two `sd`, and no granule moves as a capability.

**Scope.**
- In scope: `llvm.memcpy`, not volatile and not `.inline`, of exactly 16 bytes, both operands in the capability
  address space and known 8-aligned but not both 16-aligned, and the source not constant memory.
- 32 bytes and more are left as calls. With both cursors 8 mod 16, the middle 16 bytes are a whole granule on both
  sides, and memcpy copies it as a capability; a word copy would strip it. A larger size needs a third arm: head word,
  granules, tail word.

**Why it matters.** On FR gcc train01 (twin, shift-1 capmix), FR calls memcpy 4.06 M times against riscv-base's
1.68 M. riscv-base inlines the extra 2.4 M as `ld/ld/sd/sd`:
- bitmap_copy 1.13 M, bitmap_elt_ior 1.12 M;
- tree_to_double_int 60 k, bitmap_ior_into 47 k, bitmap_and_compl 23 k.
Each is a constant 16-byte copy between 8-aligned fields.

## The directed test (`c8-copy.c`, `run-c8.py`)

**What it does.** One virtual context on the kit's QEMU. Each case runs the copy under test, and the runtime's memcpy
through a volatile pointer, on identically prepared buffers of tagged capabilities. It requires:
- the same bytes and the same tag in every granule;
- the tags the granule geometry predicts.

**Cases.**
- Offsets: both cursors 16-aligned; dst 8 mod 16; src 8 mod 16; both 8 mod 16.
- Sizes: 16 (the pass's scope) and 32 (left a call).
- An exact self-copy (`d == s`), aligned and 8 mod 16, matched against memcpy's own self-copy.

Tags are read with `__builtin_capstone_cap_get_tag`, never inferred from values.

**Controls.**
- A plain-word copy must lose the tag.
- `-DC8_MUTANT` makes `copy16` always copy words, as a dispatch with a broken capability arm would. That build must
  fail the aligned 16-byte checks.

```
python3 run-c8.py --adapter <kit>/adapter --qemu <kit>/qemu-system-riscv64 --images <kit>/images \
  --application c8-copy.dom --work <new dir>
```

## Results (2026-11-14, llvm-c8 = f9217feb4d06 + this change; kit spec/kit-prof)

The arms are FR's configuration: `-capstone-fold-global-lo`, `+relax`, padded link script.

| arm | c8-copy | hosted | fp-s2 | malloc / nodes / w55-malloc / w55-nodes |
|---|---|---|---|---|
| C8 off | 20/20 PASS (`copy16` is a `memcpy` tail call) | 93/93 | 10/10 | failing sets identical to gate-fold's c6fr control |
| C8 on | 20/20 PASS (`copy16` is the inline dispatch) | 93/93 | 10/10 | identical to the same control |
| C8 on, `-DC8_MUTANT` | FAIL 16/20: exactly `both_aligned_16_{matches_memcpy,tags}` and the two summary checks | | | |

- The "C8 on" hosted image differs from the "C8 off" one, so the pass fired in it, and it still passes 93/93,
  including `memcpy_constant_size_unknown_alignment`.
- The malloc and node images are byte-identical across the two arms.
- lit: `CodeGen/Capstone` and `MC/Capstone`, 169 tests, 168 pass. The one failure is `shared-patches-present.test`,
  whose manifest is stale on this base (IVUsers, LICM, SeparateConstOffsetFromGEP), as it was on llvm-c7.
- `memcpy-align-dispatch.ll` checks both arms and the cases the pass leaves alone. Its ON checks fail without the flag.
- Dynamic cost per copy: `mv, mv, or, andi, beqz` plus two loads, two stores and a return or continue. That replaces a
  call of about 40 instructions. Each `ptrtoint` of a capability lowers to a `mv`.

## Self-copy cases (added 2026-11-14, after the table above)

24 checks, kit spec/kit-prof:
- C8 on (llvm-c8 FR configuration, and llvm-c9 FRCDK configuration): 24/24.
- C8 off: 24/24.
- `-DC8_MUTANT`: FAIL 18/24, exactly `both_aligned_16_*` and `self_aligned_*` plus the two summary checks.

The runner's default `--runtime` path pointed one level too shallow (`capstone/tests/runtime/virtual`). The first run of
these cases read no checks and FAILED, as a run without data must. It is fixed, and the fix is checked by a default-path
run (24/24). The table above came from the gate's copy of the runner with an explicit runtime path.

## Twin results (native twin, shift-1 capmix3, nat-fair stages as for arm FR; 2026-11-14)

Arm FRM is FR plus `-capstone-memcpy-align-dispatch`, built by llvm-c8. Arm FRMx is llvm-c8 with C8 off; its gcc
objects are identical to FR's in 386 of 387, and the one that differs is toplev.o, which embeds the compiler version.
FRM differs from FR in 19 gcc objects.

| row | outputs | user instructions, FR -> FRM | memcpy entries | image |
|---|---|---|---|---|
| 502.gcc_r train01 | identical (5 files) | 12.894 G -> 12.838 G, -56.1 M (-0.44 %) | 4,064,802 -> 1,585,752 | 6e5e26eb8e538075 (FR 1d2d347b47dbc0c6) |
| 500.perlbench_r test-sw | identical (13 files) | +1.65 M; image window equal (3,094,807,145) | | byte-identical to FR's (804f0f6074a4d2d8) |
| 505.mcf_r test | identical (6 files) | -12.9 k | | byte-identical to FR's (b85ebf872e28ccfd) |

- On gcc, memcpy itself saves 76.3 M. The inlined callers grow by part of that: bitmap_elt_ior +11.1 M, bitmap_copy
  +6.1 M.
- perl and mcf have no candidate copy: their images are byte-identical to FR's. Why each copy was left alone was not
  enumerated. Perl's whole-run delta, with an identical image, is run-to-run noise in non-image user code (the
  runtime).
