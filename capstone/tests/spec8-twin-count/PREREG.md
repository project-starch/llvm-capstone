# B5TL1's eight SPEC test stages on the twin, flags off against pool + LSR run-time endpoint + hoist (pre-registration, 2026-11-26)

The runtime lane asked for this before any board time. Both flags change codegen in every program, not only imagick,
which they were built for. A regression elsewhere, like HorizontalFilter's short loops under the unhoisted LSR option,
is what this run would show. The same run counts the hoisted LSR variant on imagick (-lsr-capability-exit-end-hoist,
3eac93f488ec).

Both options are Capstone-only codegen, so the lead's rule exempts them from the Plain arm.

## Arms

Each arm is built by perf/cclever/build-arm.sh from the same SPEC build records as B5TL1, with B5TL1's flags and
capstone-cc. The only change to the SDK is cc = llvm-sinkland clang at the lane head (3eac93f488ec).

| arm | extra options | |
|---|---|---|
| LSHOFF | none | the control |
| LSHON | -capstone-pool-base-at-entry -lsr-capability-exit-runtime-end -lsr-capability-exit-end-hoist | |
| lsrh (imagick only) | -lsr-capability-exit-runtime-end -lsr-capability-exit-end-hoist | |

Stripped sha256 (first 16 hex digits):

| program | LSHOFF | LSHON |
|---|---|---|
| perlbench_r | f3c8c80d71844564 | d1b6373ff014d9ba |
| cpugcc_r | 9908933454d63036 | e9ad9f907729139f |
| mcf_r | a1daec395a9e7260 | 92bd6c83f94070e9 |
| lbm_r | 0d9105dab2bccdfe | 184ec4f85ca8bb3e |
| x264_r | f686fe9679b04d69 | 6154a3b990b9c41f |
| imagick_r | a666e331f10d73cc | 06a0668a4535e67b (lsrh: 873462e6c4e97883) |
| nab_r | da57e5473a33e6fd | 27a7133d719bef1f |
| xz_r | ecb70755f9ee1690 | 69fdbd33b2ba851d |

**LSHOFF equals B5TL1's board images for seven of the eight programs. cpugcc_r differs, and that is why LSHOFF, not
B5TL1, is the control.** Of gcc's 387 objects, 385 are identical apart from .comment. The other two:
- toplev.o embeds __VERSION__, the compiler's revision string;
- mcf.o has one swap of two instructions after a call.

The swap persists with -disable-lsr under both compilers, so it does not come from the LSR code. B5TL1's compiler,
llvm-c9 at 29027126f39b, lacks several dev commits that the lane has. Which commit causes it is not shown.

## Procedure

Two boots, one per arm: LSHOFF first, then LSHON. Platform as the imagick loop-shape counts:
- qemu-crv ea5c05f31d93feb2, fw-y, kernel 749fed2606e3ebca (pinned copy), -m 963M, sublet.dom_binfmt=1;
- spec/prof/capmix3.so (28cd91878951c61c) at shift 4, whose SNAP tables come from a marker instruction.

Stages and their order:
- B5TL1's board stages (b5b2spec/board-stages), bin/native replaced by the arm's images, in the order lbm, mcf, nab, xz,
  gcc, imagick, (LSHON only: imagick-lsrh), x264, perlbench;
- x264's ldecod line is replaced by a cp of the host-decoded input, as in the CREVOKE scan boot;
- every command is bracketed by mark1/mark2, as in spec/prof/prof-stage.sh;
- each stage is killed 10,800 s after it starts. A stage cut in either arm is void.

Reading:
- A command's count is the mark2 table minus the mark1 table, summed over the buckets inside its executable's image
  range: from the base its "enters" line reports to the ELF's loaded span.
- Only bin/native commands are read; the arms share every other command.
- A stage's count is the sum over its native commands.
- A native command with no SNAP pair, no enters line, or a zero count is an error, never a zero.

## Predictions

1. **Control: LSHOFF imagick reads 94,483,553 ± 1,000**, the imagick loop-shape count of the same image. The argv and
   program name differ, and those moved 44 instructions last time. This checks the reader before any other row is
   read.
2. **Outputs: every stage's @@SPEC-HASH set in LSHON equals LSHOFF's.** The lsrh stage's set also equals them. imagick's
   test_output.tga is 3f70992184b213de. A stage whose outputs differ is void, whatever it saves.
3. **No stage of LSHON is worse than LSHOFF by more than +0.05 %** (the runtime lane's band), all seven non-imagick
   stages included. No direction is predicted for those.
4. **imagick, LSHON against LSHOFF: a reduction between 2.55 M and 2.85 M** (2.7 % to 3.0 %). That is pool's 1.10 M
   plus the hoisted LSR's below, less overlap.
5. **imagick, lsrh against LSHOFF: a reduction between 1.50 M and 1.75 M** (1.6 % to 1.85 %; unhoisted it was 0.209 M).
   The mechanism, read from the disassembly:
   - HorizontalFilter's inner loop is 29 instructions (off 30), and its preheader one fewer than off's (the counter's
     initial value). The endpoint stays in a register and the select runs once per row.
   - That makes -1,075,200 - 288,000 in HorizontalFilter, less at most ~20 instructions per row (600 rows).
   - NegateImageChannel is identical to the unhoisted lsr image: -286,560.
   - VerticalFilter: -26,400 in the loop, its preheader not separately predicted.

If 2 and 3 hold, the board form is one extra image: B5TL1 with both flags (LSHON), all eight stages, at the end of
board 3's b42c8c644 session after ldcexc2, against that session's B5TL1 rows. Its cpugcc_r carries the swap above as
well as the flags.

## Not covered

- Cycles. These are instruction counts on the trap twin.
- Workloads other than the SPEC test stages.
- P1, where the run-time endpoint does not hold.

## Amendment 1, before any count was read: the board form, and the hoist's scope

The semantics owner ruled on both, after the pre-registration was pushed and while the first boot waited for the lock.

**The board form is two images, not one.** LSHOFF is not B5TL1, so the board comparison is LSHOFF against LSHON. Both
were built by the lane's compiler at 3eac93f488ec, so the cpugcc_r mcf.o swap above is in both arms. Both images run all
eight stages at the end of board 3's b42c8c644 session, after ldcexc2, if predictions 2 and 3 hold. They are not
compared against the session's B5TL1 rows.

**The hoist is accepted for b42c8c644 and later Sublet seals, with P1 excluded,** on a stronger reason than the one its
comment first gave:
- CIncOffset's one trap on a tagged source, cause 26 for a sealed or uninitialised reference, is unreachable under the
  Sublet format, because no register can hold such a reference;
- SEAL and CALL, SEALEDRET's producer, are illegal (core/decoder.sv:1324-1325, :1456);
- every loaded capability decodes as NONLIN (core/include/ariane_pkg.sv:1014);
- the 26 is at capstone_flu_unit.anvil:43, :80 and :120.
These were read at capstone-ariane b42c8c644. On P1, which has types, the hoist could add a 26 where the original has
none. The comment now says this. The change is to the comment only, so the arms' images are unchanged.
