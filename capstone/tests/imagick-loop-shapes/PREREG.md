# imagick's loop shapes on the twin: the two flagged fixes counted (pre-registration, 2026-11-25)

The runtime lane asked for the two loop shapes the item-3 census found in B5TL1's imagick
(docs/history/25-11-2026_23-02-39_line-b-imagick-filter-loop-census-and-mcf-split-counter.md): the cause of each, a
fix behind a flag (default off), lit, byte-identity with the flag off, and the twin instruction count. Board time comes
only for a fix that shows 0.2 % or more on a row.

| option | shape | cause (in the commit) |
|---|---|---|
| -capstone-pool-base-at-entry | the filter weight reloaded with fld every iteration | Early MachineLICM: "Can't remat / high reg-pressure" on PseudoLLA |
| -lsr-capability-exit-runtime-end (amendment 1; was -lsr-capability-exit-inbounds) | an integer exit counter beside the capability IV | replaceCapabilityLatchTest's max-count guard: max BTC 2^64 - 2 against a 60-bit period |

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

## Amendment 1, before the run: the LSR option replaced, after the semantics ruling

The semantics owner ruled (2026-11-25) that inbounds alone is not an acceptable basis on Capstone. C code here
computes out-of-object pointers often, and inbounds is a frontend promise, not a hardware fact. The proposed basis was
a load or store through the anchor that dominates the latch. The candidate filter already requires exactly that.

That basis is not enough either, so the option was redesigned rather than retested.
- The endpoint is computed from the count, not from an accessed cursor, so at or above the period it can alias an
  early cursor before any out-of-bounds access happens.
- cursor-exit.ll's @repeating_cursor shows it. Its access dominates the latch, the count is 2 and the step 2^63, so
  the real endpoint is the second cursor. The cursor test then exits after one iteration, where the original traps at
  its second load.
- The same holds for an unbounded count of 2^61 + 1 on an 8-byte step over a 16-byte object. The original traps at
  its third load; the cursor test exits after the first.

The replacement, -lsr-capability-exit-runtime-end, chooses the endpoint in the preheader:
- the real one when the backedge count is below the period;
- otherwise the start cursor plus one byte, which no cursor of an even step reaches, so the loop leaves only by
  trapping.

The original traps too in that case: its anchor is accessed on every iteration, and a full cycle of cursors spans
2^64 - 2^ctz(step) bytes, so one of them lies at or above 2^38. Accessed cursors lie in [0, 2^38) by the format:
a reference with a cursor at or above 2^38 is written untagged and traps at its access (capstone-qemu
target/riscv/cap.h:199-233, sublet-reference-model.md section 2.1). An earlier draft of this amendment said the basis
was that no capability's bounds reach 2^63 bytes; that is false, since Sublet bounds are not limited. The semantics
owner gave the cursor rule. Inside the loop nothing changes. The cost is a compare and a select in the preheader.

The test, cursor-exit-runtime-end.ll, covers:
- the three pinned shapes, which keep their flag-off pins in cursor-exit.ll: @unbounded gets the run-time choice, and
  @repeating_cursor and @bound_bad the constant unreachable endpoint;
- a statically bounded loop, which gets no run-time choice;
- two negative controls that keep the integer test: an access under a condition, and an anchor without inbounds.
Its ON checks fail against the option-off output.

Lit: CodeGen/Capstone, LoopStrengthReduce and LICM pass with both options off.

The images were rebuilt with the revised compiler:

| variant | stripped sha256 |
|---|---|
| off | a666e331f10d73cc (unchanged; still byte-identical to B5TL1's board image) |
| pool | f8dfaf92927c6db8 (unchanged) |
| lsr | 4b71218d648689cf (was 56e3385c05ca42fc) |
| both | 2d2220ac2417d52c (was 6d61d2d5924e02c8) |

The inner-loop sizes are unchanged (lsr 29 / 31 / 11, both 28 / 30 / 11), so the predictions stand as written. The
preheader compare and select run once per loop entry, and the bounds in prediction 4 cover them.

## Amendment 2, after the first boot and before any count was read: one boot per variant

The first boot (2026-11-26 00:17, 11 s after the lock) ran all four variants. Each loads at the same base,
0x3ff7c00000, because they are builds of one image. So the single FINAL table sums their per-pc counts, and no
variant's count can be read from it. Nothing from that table is read.
- The item-3 profile separated its processes by load address only because its images differed; that was not
  rechecked when four builds of one image went into one boot.
- What that boot does settle is **prediction 2: all four test_output.tga hashes are equal**
  (3f70992184b213de3b457de27312f52f1222877fd5bb17eeb8e1e9c67a1b0deb), and every run exited rc 0.

The counts come from four boots, one variant each, in the order off, pool, lsr, both. The images, the command,
the platform and the predictions are unchanged. Each boot's FINAL table holds one imagick process; the image range
is separated from the shell and busybox at other addresses, as before.
