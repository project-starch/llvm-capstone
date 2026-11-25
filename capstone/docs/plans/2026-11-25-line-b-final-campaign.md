# Line B: finishing the bitstream and the final measurement campaign

Status: plan, 2026-11-25. Owner: the runtime lane. It continues the post-submission programme on `spec-silicon`
(`capstone/docs/plans/2026-11-20-post-submission-programme.md`, which has not yet landed on dev). It narrows to what
Line B needs before the final campaign, the last item of the programme by the lead's rule: SPEC train runs only at
the very end.

## Where Line B stands (each row is cited where it was read)

| Item | State | Record |
|---|---|---|
| B5b-2, the trap route on the Sublet format, 904cd9c6b | CONFIRMED N=2, bare kits | b5b2 plan, dev |
| Linux on B5b-2 with the keep-tags firmware | boots; regs.S 0 (strip control 23); the stock firmware faults with 24 | `tests/rtl-smoke/b5b2-linux-2026-11-25/` (634fd5531e99) |
| Interruptible CREVOKE 4(b), plus the store-queue drain and the CAP_WB guard | seal 88db40027, 16 arms plus the control as predicted (8 discriminating), stop 1,130 cycles; N=1 | d139d076f84e |
| One-operand applications on B5b-2 | five, with stock's answers | nested-workloads ba26257f256a |
| SPEC test on B5b-2 | gcc passes; x264 and imagick fault 24 with tval 0 (imagick a NULL increment; x264 consistent with one; the untagged-CINCOFFSET rule is the explanation that 28fcbb646 tests); mcf wedges | `tests/rtl-smoke/b5b2-spectest-2026-11-25/` (runtime/b5b2-linux) |
| The FLU ungate (reference-exact untagged CINCOFFSET, CINCOFFSETIMM, SCC) | in the combined build 28fcbb646, not yet sealed | land/rtl-crevoke4b |
| The mcf wedge | open: an LDC at the LSU head, four committed stores without a D$ grant, at least 1M cycles; M-16's LSU-head signature (sw=219/220), a different store port (sw=221) | the 4(b) plan; board/meas-pw 57c5f32e3880 |

## Before the final bitstream

1. **The mcf wedge.** First rtl's directed sim hunt. Then the aperture build (part of 28fcbb646), with the mcf image
   re-run to a silence-stop snapshot. Per-candidate predicted readings are written first, and two candidates must not
   read alike. "Final" waits for a fix or a localisation.
2. **The combined build 28fcbb646.** Its synthesis-only gate first, with the FLU result path under the 40 GB guard,
   then the full build and the seal. On the seal: the bare kits (b5b2n, crevoke4b, wapflu) at N=2 on two boards,
   Linux boot 3's image, and the SPEC test chain re-read. Predictions: x264 and imagick pass in all three arms; every
   other stage reads as on 904cd9c6b, within ±0.5 % of its time where both passed (the drain's cost is about 1.3 cycles
   per lifetime request).
3. **Stage 2** (compressed capability instructions) is out of the final bitstream by default. That is pending the
   lead, and compiler writes the decoder delta as a spec only.

## The arms of the final campaign

| Arm | Bitstream | Firmware | Kernel | Userland |
|---|---|---|---|---|
| Full (Line B) | the final Line B seal | keep-tags (eb85e21c or later) | Linux 6.1 + the trap-route series, NO_HZ_IDLE at HZ=100 | B5TL1: compiler llvm-c9, the trap runtime with alloc-lean's mallocng series and the CRENEW mapping cache (runtime/trap-alloc-lean) |
| Plain | rvbase 8db8f9cda (`caplifive_rvbase-matched-ic32k-dtlb64`), 32K 8-way L1D, the same as Line B's | stock OpenSBI 1.3.1, the same tree with the flag off | Linux 6.1, the same board config edits, NO_HZ_IDLE at HZ=100, no Capstone series (`trapk/plain/build-kernel-plain.sh`) | native RISC-V images with the same mallocng policy and no lifetime operations, built by the same compiler and flags apart from the Capstone ones |
| B5TL0 (an ablation, optional) | as Full | as Full | as Full | B5TL1 without mapping reuse |

**The matching rules,** each to be checked from the artifact and not from the recipe:
- **Tick.** The bring-up kernel is HZ_PERIODIC, so the final Full kernel moves to NO_HZ_IDLE at HZ=100, the measured
  configuration since PREREG Addendum 66. The bring-up chain stays as built and is labelled bring-up.
- **THP, SMP, KSM and modules** off on both arms: the trap-route kernel requires it (one possible CPU, no THP).
- **The DTS memory node** equal on both arms (`reg = <0x0 0x80000000 0x0 0x3c2d2000>`); read the Plain kit's DTB before
  the first Plain boot.
- **The L1D and I-cache** equal (32 KiB, 8 ways, 128-bit lines), from each bitstream's config package, as the gap audit
  read them for 88db40027, 28fcbb646 and 8db8f9cda.
- **The compiler identity** is printed beside every number.

## Still to build

- The Full NO_HZ_IDLE kernel: the same series and recipe as `b5b2-linux-2026-11-25/build/build-kernel.sh` plus the tick
  edit.
- The Plain SPEC images paired with B5TL1 (compiler), and the Plain test rows on rvbase, pre-registered (board).
- **Train stages for Line B.** Inputs over the 26 MB image cap need an external initrd, which the trap payload lacks
  (its initramfs is built in). Before any train boot, run a twin memory check per input at 963 MiB, and find the trap
  kernel's table-size knob (the native route's 2^20 / 2^22 record ceilings for Perl and gcc have no recorded
  trap-route equivalent).
- **+Spatial on Line B:** the spatial-only core exists only on P1. Whether the final +Spatial row needs a Line B core
  is decided before the campaign. If it does, rtl ports step C's parameter.
- The re-registrations named by Addendum 236 (Addendum 231, the ordering half of 232, t202s2's two rows), restated
  for the Line B arms.

## The campaign (last)

Three runs per arm, board-constant pairs, drift-balanced order, intervals. SQLite --size 100 and the Fig. 10 Perl
suns.pl row belong here too, since both are long or train inputs.
