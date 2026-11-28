# imagick's loop shapes on the twin: results (2026-11-26)

Pre-registration: PREREG.md, 9a2c32d47972 / 99f138ebace6, with amendment 1 (544b3b939631) and amendment 2
(f018d75de135), all pushed before any count was read.

**-capstone-pool-base-at-entry saves 1,099,810 instructions (1.164 %) on B5TL1's imagick, every prediction for it
holding. -lsr-capability-exit-runtime-end saves 209,447 (0.222 %): its loop savings are exactly as predicted, but in
HorizontalFilter the preheader it adds costs more than the loop saves, so its whole-image prediction fails.** Both
clear the 0.2 % bar; the LSR option only just, and only through NegateImageChannel.

## The boots

Four boots, one variant each, 2026-11-26 00:31 to 01:28. Platform as pre-registered: qemu-crv ea5c05f31d93feb2, fw-y,
kernel 749fed2606e3ebca, -m 963M, capmix3 at shift 1, one FINAL table per boot. Every run exited rc 0. Images as in
amendment 1: off a666e331f10d73cc, pool f8dfaf92927c6db8, lsr 4b71218d648689cf, both 2d2220ac2417d52c.

| variant | image instructions | reduction | % | HF / VF / Negate inner loop |
|---|---:|---:|---:|---|
| off | 94,483,553 | | | 30 / 32 / 12 |
| pool | 93,383,743 | 1,099,810 | 1.164 | 29 / 31 / 12 |
| lsr | 94,274,106 | 209,447 | 0.222 | 29 / 31 / 11 |
| both | 93,181,106 | 1,302,447 | 1.378 | 28 / 30 / 11 |

No counted pc in any boot fell outside an instruction start (non-start 0).

## Predictions

1. **off = 94,483,597: FAILS, by 44.** Every one of the 44 lies in string and path code fed by the command line
   (strlen +56, GetPathComponent +36, CopyMagickString +35, ParseCommandOption +26, strchrnul +24, memcpy -91, and
   others): the item-3 boot ran the image as imagick-b5tl1, this one as imagick-off. No filter function differs. So
   the boots are comparable with item 3 to within 44 instructions; the same libc functions move by up to 38 between
   these four boots, which sets the cross-boot floor at about 10^2, far below every effect read here.
2. **Every test_output.tga equal to off's: HOLDS.** All four read 3f70992184b213de, in the first boot and again in
   each of the four.
3. **Loop savings: HOLDS exactly** for all three (pool 1,101,600, lsr 1,389,600, both 2,491,200), read as inner-loop
   size times the item-3 iteration counts.
4. **Whole-image lower bounds: pool HOLDS (1,099,810 ≥ 1.00 M); lsr FAILS (209,447 < 1.25 M); both FAILS
   (1,302,447 < 2.3 M).**
5. **≥ 0.2 % (189 k): HOLDS** for all three.

## Reading prediction 4, per function (as the pre-registration says it must be read)

Per function, against off. The histograms are of execution counts inside the function, so a change can be placed by
how often it runs.

| function | pool | lsr |
|---|---:|---:|
| HorizontalFilter | -1,073,996 | +80,400 |
| VerticalFilter | -25,436 | -3,200 |
| NegateImageChannel | 0 | -286,560 |
| every other function | -378 | -87 |

**pool.** HorizontalFilter loses one instruction at 1,075,200 executions and gains 4 run once (the entry-block base
pair) and 2 at 600; VerticalFilter the same shape at 26,400 / 1 / 480. The shear functions gain from other hoisted
pool loads (-731), QueryColorCompliance and its kin pay the entry pair (+306).

**lsr.**
- **NegateImageChannel**: -1 at 288,000 executions, +3 at 480. Its loop is entered 480 times at 600 iterations each,
  so the preheader is paid for at once.
- **HorizontalFilter**: -1 at 1,075,200 and **+4 at 288,000**, +6 at 600. The inner loop is entered 288,000 times and
  runs 3.73 iterations on average, so 4 instructions per entry outweigh 1 per iteration.
- **VerticalFilter**: -1 at 26,400, -1 at 1,760, +3 at 7,200, +7 at 480; 3.67 iterations per entry, a small net gain.

The +4 per entry in HorizontalFilter, read from the disassembly (lsr, the 288,000-count block before the loop):
- `srli a7, a5, 60` and `beqz a7`: the count-below-period test. The select became a branch.
- `ld a7, 0x130(sp)` and `cincoffset a7, s7, a7`: the real endpoint, start plus 16 * count, with 16 * count spilled.
- `li a7, 12`: a constant off kept in ra, rematerialised here.
- less `li a6, 0`: the integer counter's initial value.

Every input to the endpoint and to the test (s7, the count s0, a5 = s0 - 1) is set at the 600-count level, once per
row, so all of it is invariant in the loop over x that enters the inner loop. It is emitted in the inner loop's
preheader because replaceCapabilityLatchTest builds it there. LSR runs after IR LICM, and nothing later hoisted
the branch the select became; why MachineLICM left it is not shown.

## What this does not cover

- Other SPEC rows. Only imagick was rebuilt and counted.
- Cycles. These are instruction counts on the trap twin; the board decides time.
- The LSR option's net is entry-count dependent: a loop entered often for few iterations loses. A version that
  emits the endpoint at the outermost loop where its operands are invariant is the obvious next variant, and would
  need its own pre-registration.

## Instrument identity

- Compiler: llvm-sinkland, the lane's head code (clang reports 55066522c370; 80136543f163 after it changes only a
  comment). The counted lsr and both images were built at 00:09, a minute before bc38212981f2 (the covering-access
  requirement) was committed, and pool at 23:46, before the LSR option was replaced. All four were rebuilt with this
  compiler after the count, and each full ELF is identical to the counted one: off bc421f6de8b7bc2e (B5TL1's own),
  pool 3a4197ab312041d2, lsr 2dfba4cec525dc0a, both 58688202417cd8b7.
- Reader: count.py (the per-variant table reader), controlled on the item-3 table: 30 / 32 / 12 and 94,483,597,
  exact. The run's scripts stay with the run's local files, because they carry host paths.
