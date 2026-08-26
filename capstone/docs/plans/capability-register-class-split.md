# Split the register class: capabilities get their own, integers get theirs back

Status: PLAN, not started. Prerequisite for `drop-i128`.
Branch base: `capstone-captype` (merged state, 59/59 lit, core tier verified).

## Why this, before drop-i128

Removing `addRegisterClass(MVT::i128, &Capstone::GPRRegClass)` was measured on
2026-08-24: Capstone lit falls from 59/59 to **40/59**, and all 19 breaks are
type-legalisation ("Unexpected custom legalisation", "Do not know how to expand",
`NVT.bitsGE(VT)`). They are not 19 independent bugs. They are one fact: i128
cannot be split into two i64 halves the way every other RV64 target splits it,
because on this target i128 has a *register*.

It has a register because a capability has to live somewhere, and capabilities
share `GPR` with integers. So the ordering is forced: split the class first,
then i128 stops being special and the generic splitter does the work.

CHERI does exactly this and it is worth stating what they do NOT have to do as a
result: no `lowerScalarI128*` family, no `PseudoSCALAR_COPY_I128`, no
`PseudoTRUNC_CAP`, no operand-role guessing. `MVT::i128` is never a legal
register type in their RISC-V backend.

## The root fact, measured

    CapstoneRegisterInfo.td:225  XLenRI : RegInfoByHwMode<[RV32, RV64],
                                   [RegInfo<32,32,32>, RegInfo<128,128,128>]>

**A Capstone integer GPR claims to be 128 bits wide on RV64.** It is not; it is
64. The declaration is 128 because the same register may hold a capability.

That single lie is upstream of most of what this fork has had to work around,
and it also means the 276 uses of `GPR` in CapstoneInstrInfo.td -- nearly all
inherited from upstream RISC-V, where GPR is XLen-wide -- are quietly operating
on a class that is twice the width they assume. Splitting the class does not
just enable drop-i128; it makes those 276 uses correct again.

CHERI, for comparison:

    XLenRI : [RegInfo<32,32,32>, RegInfo<64,64,64>]      // integers
    CLenRI : [RegInfo<64,64,64>, RegInfo<128,128,128>]   // capabilities
    GPR  : RegisterClass<..., [XLenVT, XLenFVT, i32], ...> { let RegInfos = XLenRI; }
    GPCR : RegisterClass<..., [CLenVT], ...>             { let RegInfos = CLenRI; }
    def sub_cap_addr : SubRegIndex<-1, -1>;
    class RISCVCapReg<RISCVReg subreg, ...> { let SubRegs = [subreg];
                                              let SubRegIndices = [sub_cap_addr]; }

## Shape of the change

**Keep the X names.** Add `C0..C31` as new 128-bit registers whose
`sub_cap_addr` subregister is the existing `X0..X31`, and shrink `XLenRI` to 64
bits. The X registers keep their names and their 276 pattern uses, and those
uses become correct rather than needing edits. Only the 14 capability
instructions move to the new class.

This is the incremental direction. The alternative -- redefining what `X11`
means -- changes 276 sites at once and cannot be measured in stages.

Existing precedent in the file: `X11 -> X11_W -> X11_H` already form a
subregister chain with `sub_32`/`sub_16`, so the machinery is present and
understood here. The new level goes on TOP rather than below.

## Surface area, counted 2026-08-24

| what | count |
|---|---:|
| `GPR` in CapstoneInstrInfo.td | 276 (stay as-is; they are integer uses) |
| `GPRRegClass` in target C++ | 72 (audit; the capability ones move) |
| GPR subclasses in the .td | 19 (each needs a decision: integer, capability, or both) |
| capability instructions | 14 (LDC, STC, CIncOffset(Imm), MOVC, LCC, SHRINK, TIGHTEN, SEAL, INIT, DELIN, SCC, MREV, REVOKE) |

## Stages, each with its own measurement

Every stage ends with a number, not an opinion. The gate at each step is
Capstone lit plus the nightly quick tier; the full core tier runs before the
branch is called done.

**S1. Add the class, change nothing else.** Declare `C0..C31`, `sub_cap_addr`,
`CLenRI`, `GPCR`. Do NOT move any pattern to it yet, do NOT shrink `XLenRI`.
Expect: lit unchanged at 59/59. This is a build-and-TableGen check only, and it
is cheap. If TableGen objects here, everything after is different.

**S2. Move the 14 capability instructions to GPCR.** Their operands become
`GPCR`, `capstone_cincoffset`'s profile becomes `(c128, c128, i64)` against the
new class, LdPat/StPat for c128 use GPCR. Measure lit. The interesting failures
are the ones where a value has to cross classes.

**S3. ptrtoint / inttoptr become subregister operations.**
`(XLenVT (EXTRACT_SUBREG GPCR:$c, sub_cap_addr))` replaces `PseudoTRUNC_CAP`.
Deleting that pseudo is the point: it is an `addi rd, rs, 0`, which is
simultaneously a copy and a tag clear, and that double meaning caused the
inline-asm untagging regression on 2026-08-23. A subregister read cannot.

  NOT copyable from CHERI: their `inttoptr` is `(CIncOffset C0, GPR:$rs2)`.
  Verified against our primary source, `capstone-qemu` op_helper.c
  `helper_cscincoffset`: it raises `RISCV_EXCP_UNEXP_OP_TYPE` when rs1 is
  untagged, and cnull is untagged. Capstone needs its own construct here. This
  is the one place in the plan with no worked precedent, so it goes early.

**S4. copyPhysReg splits into two arms by register class.** GPCR copies are
tag-preserving; GPR copies are integer moves. The `CapstoneScalarCopyForLiveSrc`
flag and its whole comment block become unnecessary: the question it answers by
heuristic ("is this copy a capability?") is answered by the class.

**S5. Calling convention and frame.** Capability arguments and returns in C
registers; `SP` and the gp-captable base become GPCR subclasses; spill/reload of
a GPCR uses `stc`/`ldc`. `CapstoneCallingConv.cpp`'s ValVT tests become class
questions.

**S6. Shrink XLenRI to 64 bits.** Only after S2-S5. This is the step that makes
the 276 inherited patterns honest, and the step most likely to surface latent
assumptions.

**S7. drop-i128.** Remove `addRegisterClass(MVT::i128, ...)`, delete the six
`lowerScalarI128*` functions and the i128 arms of `ReplaceNodeResults` /
`LowerOperation`, and let the generic type legaliser split i128 into two i64s
as it does on every other RV64 target. Expect this stage to be mostly deletion.
The 19 breaks measured today should resolve here; if they do not, the count
after S6 is the real number and the estimate was wrong.

## Gates

* `capstone/tests/scan-tag-stripped-caps.py` over every built domain. It detects
  the exact failure class this work eliminates, and it has controls in both
  directions (8 hits on the CoreMark domain built while the bug was live, 0
  after). Once S3 lands it should be permanently 0, and if it ever fires again
  the answer is a bug, not a judgement call.
* The nightly quick tier per stage; the full core tier before merge.
* A baseline build of the parent branch for any suite that regresses. Twice on
  2026-08-23 this was the only thing that separated a real regression from the
  machine's boot-flake rate.

## Risks, named

* **Register aliasing is the part most likely to surprise.** C and X registers
  are the same hardware register seen at two widths. LLVM models super/sub
  aliasing natively, but allocation pressure, live-range splitting and spill
  placement all change at once, and none of that is visible in lit.
* **S1-S7 cannot each keep the tree green.** S2 through S5 are one coherent
  change to how a capability is represented; the tree will be red in between.
  Plan for a single working branch with measured checkpoints, not for seven
  mergeable commits.
* **The 19 is the first wall, not the distance.** Measured on 2026-08-24 by
  flipping the line and building. The same shape of estimate was wrong earlier
  the same day for captype: a bucket of 10 cleared and moved exactly one test to
  green, because the next layer sat behind it.

## Out of scope, deliberately

`uintptr_t` is 4 bytes on this target while a pointer is 16 (verified with a
positive control). CHERI makes `uintptr_t` capability-sized and
provenance-carrying, with `ptraddr_t` as the plain 64-bit address, which is why
their backend never has to recover provenance and ours does
(`recoverCapabilityFromAddressArith`). That is a front-end and ABI decision, it
is the project lead's, and it is not part of this plan.
