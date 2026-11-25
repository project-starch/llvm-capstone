# imagick's loop shapes on the twin: the two flagged fixes counted (pre-registration, 2026-11-25)

The runtime lane asked for the two loop shapes the item-3 census found in B5TL1's imagick
(docs/history/25-11-2026_23-02-39_line-b-imagick-filter-loop-census-and-mcf-split-counter.md): the cause of each, a
fix behind a flag (default off), lit, byte-identity with the flag off, and the twin instruction count. Board time comes
only for a fix that shows 0.2 % or more on a row.

| option | shape | cause (in the commit) |
|---|---|---|
| -capstone-pool-base-at-entry | the filter weight reloaded with fld every iteration | Early MachineLICM: "Can't remat / high reg-pressure" on PseudoLLA |
| -lsr-capability-exit-inbounds | an integer exit counter beside the capability IV | replaceCapabilityLatchTest's max-count guard: max BTC 2^64 - 2 against a 60-bit period |

The commits on compiler/imagick-loop-shapes: 9a2c32d47972 and 99f138ebace6.

The third shape, VerticalFilter's in-loop offset add, has no fix. LICM's hoistGEP refuses capability pointers on
purpose, because hoisting `p + x` forms a cursor the loop never forms. Proving that cursor accessed needs
contribution[0]'s loaded pixel. It costs 26,400 instructions (0.03 %), under the bar.

## Images

B5TL1's 538.imagick_r, rebuilt from the arm's own compile.txt and link.txt with the lane's compiler (llvm-sinkland at
99f138ebace6), every other input as recorded. Stripped as the arm strips.

| variant | options | stripped sha256 |
|---|---|---|
| off | none | a666e331f10d73cc (byte-identical to B5TL1's board image; full ELF bc421f6de8b7bc2e, the arm's) |
| pool | -capstone-pool-base-at-entry | f8dfaf92927c6db8 |
| lsr | -lsr-capability-exit-inbounds | 56e3385c05ca42fc |
| both | both options | 6d61d2d5924e02c8 |

Static inner-loop sizes, in instructions (off / pool / lsr / both), with iterations from the item-3 profile:

| loop | off | pool | lsr | both | iterations |
|---|---:|---:|---:|---:|---:|
| HorizontalFilter | 30 | 29 | 29 | 28 | 1,075,200 |
| VerticalFilter | 32 | 31 | 31 | 30 | 26,400 |
| NegateImageChannel | 12 | 12 | 11 | 11 | 288,000 |

## Procedure

One boot on the trap twin, as the item-3 profile boot:
- qemu-crv ea5c05f31d93feb2, fw-y, kernel 749fed2606e3ebca (pinned copy), -m 963M, sublet.dom_binfmt=1;
- capmix3 at shift 1, one FINAL table, the processes separated by load address;
- the four variants in the order off, pool, lsr, both, each running the SPEC test command;
- each run's test_output.tga hashed.

## Predictions

1. **off reads 94,483,597 instructions**, the item-3 reading of the same image. Otherwise the boot is not comparable
   with that profile, and only the in-boot differences are read.
2. **Every variant's test_output.tga has off's sha256.** A variant that differs is void, whatever it saves.
3. **The three loops' savings, from per-pc counts:**
   - pool: 1,101,600 (1,075,200 + 26,400);
   - lsr: 1,389,600 (1,075,200 + 26,400 + 288,000);
   - both: 2,491,200.
4. **Whole image, reduction against off:**
   - pool: at least 1.00 M (1.06 %);
   - lsr: at least 1.25 M (1.32 %);
   - both: at least 2.3 M (2.4 %).
   Outside the loops: the entry-block base adds two instructions per call of a function with a rebased pool, and other
   hoisted pool loads save some. The bounds leave room for both.
5. **All three variants clear the 0.2 % bar** (189 k instructions).

If 3 holds and 4 fails, code outside the loops costs more than the loops save. That is read per function, not
explained in advance.
