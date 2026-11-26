# Line B item 3: imagick's filter loops on B5TL1 against Plain-matched, and mcf's split counter re-read (2026-11-25)

The runtime lane asked for imagick's filter-loop hot-spot census of B5TL1 (the Line B campaign's Full arm) against
riscv-base, and for mcf's split counter re-read on B5TL1 in the same boot. Here riscv-base is the campaign's
Plain-matched image (tests/line-b-plain, on dev at 22145821659d): the same clang and generic options, static model,
round-evict mapping cache. So, unlike the C4 notes, whose riscv-base was GCC-built, this compares two clang builds.

## The boot

One boot on the QEMU trap twin.
- QEMU: qemu-crv, ea5c05f31d93feb2. Firmware: sbi-keep-run fw-y. Kernel: trapk 749fed2606e3ebca. -m 963M,
  sublet.dom_binfmt=1.
- Profiler: spec/prof capmix3.so, 28cd91878951c61c, shift=1, so one bucket is one instruction start.
- Three runs, each the stage's run command only:
  - B5TL1 imagick, a666e331f10d73cc;
  - Plain-matched imagick, b461c489b84ebfbc;
  - B5TL1 mcf, a1daec395a9e7260.
- SPEC test inputs. All three exited rc 0, and the guest powered off cleanly.

How the processes were told apart. capmix3 wrote one FINAL table, and the processes are separated by load address.
- The kernel entered imagick at 0x3ff7c00000 and mcf at 0x3ff7f00000. Their pcs map to link addresses as pc - base +
  0x10000.
- Plain-matched is static at its link address.
- Every profiled pc inside each image is an instruction start (the census counts the ones that are not, and reads 0).

Symbols come from perf/cclever/arms/B5TL1/exes/*.full. Their .text is byte-identical to the profiled stripped
images: imagick 5993b20e2e981d51, mcf 38b10966e3f83d5d.

**What this measures:** instructions only. capmix3 defines a data-access and L1D-model callback (on_mem) but never
registers it, so its access, byte and miss fields are zero in every bucket by construction. Nothing here speaks to
memory traffic or cycles.

## imagick

Whole image: B5TL1 94,483,597 instructions, Plain-matched 93,628,093, ratio 1.0091 (+855,504).

| function | B5TL1 | Plain-matched | B5TL1 minus Plain |
|---|---:|---:|---:|
| HorizontalFilter | 53,676,147 | 52,473,793 | +1,202,354 |
| memcpy | 17,948,563 | 18,316,182 | -367,619 |
| WriteTGAImage | 8,368,509 | 8,653,627 | -285,118 |
| NegateImageChannel | 3,470,489 | 3,183,439 | +287,050 |
| ReadPixelCachePixels | 2,923,875 | 3,211,035 | -287,160 |
| WritePixelCachePixels | 2,923,875 | 3,211,035 | -287,160 |
| VerticalFilter | 1,494,932 | 1,412,384 | +82,548 |
| strcasecmp | 782,215 | 602,092 | +180,123 |
| free_slot (heap lifetimes) | 116,930 | 0 | +116,930 |

The filter loops are the hot spots: HorizontalFilter is 57 % of B5TL1's instructions. Each function's hottest run of
pcs (every pc executed as often as its hottest pc) is its inner loop.

| loop | iterations | B5TL1 per iteration | Plain per iteration | the difference |
|---|---:|---:|---:|---|
| HorizontalFilter | 1,075,200 | 30 | 28 | split counter, filter weight reloaded |
| VerticalFilter | 26,400 | 32 | 29 | split counter, filter weight reloaded, offset add |
| NegateImageChannel | 288,000 | 12 | 11 | split counter |

The three shapes:
- **Split counter.** B5TL1 keeps an integer trip counter (HorizontalFilter `addi a6, a6, 1` ... `bne s0, a6`) beside
  the capability pointer it advances (`cincoffsetimm t0, t0, 0x10`). Plain-matched exits on the pointer itself
  (`addi a4, a4, 0x10` ... `bne a4, s4`). That is one extra instruction per iteration in all three loops.
- **Filter weight reloaded.** B5TL1 loads `fld ft6, 0x48(s2)` every iteration. Plain-matched holds the same double in
  ft1 from before the loop. The loop has no store, so it is a load LICM did not hoist on the Capstone side.
- **Offset add (VerticalFilter only).** B5TL1 forms the pixel address as `add t2, t2, a2` / `slli` /
  `cincoffset t2, s10, t2` inside the loop. Plain-matched folds the row offset into its base register before the
  loop (`slli` / `add a2, a2, t5`).

Otherwise the bodies hold the same instructions, up to scheduling order, with cincoffset where Plain has add.

In HorizontalFilter the two extra instructions per iteration come to 2,150,400. The function's whole excess is
1,202,354, so outside the inner loop B5TL1 executes about 948,000 FEWER instructions than Plain-matched.

## mcf (B5TL1 only in this boot)

26,644,285,988 instructions. price_out_impl is 3,453,107,476 of them (13.0 %), after spec_qsort (37.7 %),
cost_compare (22.5 %) and primal_bea_mpp (21.8 %).

The split counter, as described in 20-11-2026_21-20-00 (b):
- `mv a3, s10` at 0x1444c (the depth-2 header): 154,195,970;
- `addi s10, a3, 0x1` at 0x14624 (its latch): 154,194,120.

The copy is 154,195,970 instructions, 0.579 % of mcf's. That is the same 154 M and 0.58 % the C4 note read on its
arm, so the count does not move on B5TL1. (mcf's Plain-matched partner was not in this boot.)

## What this does and does not settle

- On the SPEC test input, B5TL1's instruction excess over Plain-matched on imagick is 0.91 %. Most of it is two
  shapes in the filter loops, the split counter and an unhoisted load, which the Capstone side's own code outside the
  loops more than offsets.
- If B5TL1's measured imagick overhead on the board is larger than about 1 %, it is not in instruction count. This
  boot cannot say where it is, since its memory fields are structurally empty.
- The causes of the two loop shapes are not established here: why LSR keeps a separate exit counter beside a
  capability IV, and why LICM leaves the fld. Each is a candidate compiler lever with a known size:
  - imagick: 1 to 3 instructions per filter iteration: 2,517,600 of 94.5 M in the three loops (2 x 1,075,200 + 3 x 26,400 +
    288,000);
  - mcf: 0.58 %.
