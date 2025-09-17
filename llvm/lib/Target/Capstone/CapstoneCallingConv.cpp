//===-- CapstoneCallingConv.cpp - Capstone Custom CC Routines ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom routines for the Capstone Calling Convention.
//
//===----------------------------------------------------------------------===//

#include "CapstoneCallingConv.h"
#include "CapstoneSubtarget.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCRegister.h"

using namespace llvm;

// Calling Convention Implementation.
// The expectations for frontend ABI lowering vary from target to target.
// Ideally, an LLVM frontend would be able to avoid worrying about many ABI
// details, but this is a longer term goal. For now, we simply try to keep the
// role of the frontend as simple and well-defined as possible. The rules can
// be summarised as:
// * Never split up large scalar arguments. We handle them here.
// * If a hardfloat calling convention is being used, and the struct may be
// passed in a pair of registers (fp+fp, int+fp), and both registers are
// available, then pass as two separate arguments. If either the GPRs or FPRs
// are exhausted, then pass according to the rule below.
// * If a struct could never be passed in registers or directly in a stack
// slot (as it is larger than 2*XLEN and the floating point rules don't
// apply), then pass it using a pointer with the byval attribute.
// * If a struct is less than 2*XLEN, then coerce to either a two-element
// word-sized array or a 2*XLEN scalar (depending on alignment).
// * The frontend can determine whether a struct is returned by reference or
// not based on its size and fields. If it will be returned by reference, the
// frontend must modify the prototype so a pointer with the sret annotation is
// passed as the first argument. This is not necessary for large scalar
// returns.
// * Struct return values and varargs should be coerced to structs containing
// register-size fields in the same situations they would be for fixed
// arguments.

static const MCPhysReg ArgFPR16s[] = {Capstone::F10_H, Capstone::F11_H, Capstone::F12_H,
                                      Capstone::F13_H, Capstone::F14_H, Capstone::F15_H,
                                      Capstone::F16_H, Capstone::F17_H};
static const MCPhysReg ArgFPR32s[] = {Capstone::F10_F, Capstone::F11_F, Capstone::F12_F,
                                      Capstone::F13_F, Capstone::F14_F, Capstone::F15_F,
                                      Capstone::F16_F, Capstone::F17_F};
static const MCPhysReg ArgFPR64s[] = {Capstone::F10_D, Capstone::F11_D, Capstone::F12_D,
                                      Capstone::F13_D, Capstone::F14_D, Capstone::F15_D,
                                      Capstone::F16_D, Capstone::F17_D};
// This is an interim calling convention and it may be changed in the future.
static const MCPhysReg ArgVRs[] = {
    Capstone::V8,  Capstone::V9,  Capstone::V10, Capstone::V11, Capstone::V12, Capstone::V13,
    Capstone::V14, Capstone::V15, Capstone::V16, Capstone::V17, Capstone::V18, Capstone::V19,
    Capstone::V20, Capstone::V21, Capstone::V22, Capstone::V23};
static const MCPhysReg ArgVRM2s[] = {Capstone::V8M2,  Capstone::V10M2, Capstone::V12M2,
                                     Capstone::V14M2, Capstone::V16M2, Capstone::V18M2,
                                     Capstone::V20M2, Capstone::V22M2};
static const MCPhysReg ArgVRM4s[] = {Capstone::V8M4, Capstone::V12M4, Capstone::V16M4,
                                     Capstone::V20M4};
static const MCPhysReg ArgVRM8s[] = {Capstone::V8M8, Capstone::V16M8};
static const MCPhysReg ArgVRN2M1s[] = {
    Capstone::V8_V9,   Capstone::V9_V10,  Capstone::V10_V11, Capstone::V11_V12,
    Capstone::V12_V13, Capstone::V13_V14, Capstone::V14_V15, Capstone::V15_V16,
    Capstone::V16_V17, Capstone::V17_V18, Capstone::V18_V19, Capstone::V19_V20,
    Capstone::V20_V21, Capstone::V21_V22, Capstone::V22_V23};
static const MCPhysReg ArgVRN3M1s[] = {
    Capstone::V8_V9_V10,   Capstone::V9_V10_V11,  Capstone::V10_V11_V12,
    Capstone::V11_V12_V13, Capstone::V12_V13_V14, Capstone::V13_V14_V15,
    Capstone::V14_V15_V16, Capstone::V15_V16_V17, Capstone::V16_V17_V18,
    Capstone::V17_V18_V19, Capstone::V18_V19_V20, Capstone::V19_V20_V21,
    Capstone::V20_V21_V22, Capstone::V21_V22_V23};
static const MCPhysReg ArgVRN4M1s[] = {
    Capstone::V8_V9_V10_V11,   Capstone::V9_V10_V11_V12,  Capstone::V10_V11_V12_V13,
    Capstone::V11_V12_V13_V14, Capstone::V12_V13_V14_V15, Capstone::V13_V14_V15_V16,
    Capstone::V14_V15_V16_V17, Capstone::V15_V16_V17_V18, Capstone::V16_V17_V18_V19,
    Capstone::V17_V18_V19_V20, Capstone::V18_V19_V20_V21, Capstone::V19_V20_V21_V22,
    Capstone::V20_V21_V22_V23};
static const MCPhysReg ArgVRN5M1s[] = {
    Capstone::V8_V9_V10_V11_V12,   Capstone::V9_V10_V11_V12_V13,
    Capstone::V10_V11_V12_V13_V14, Capstone::V11_V12_V13_V14_V15,
    Capstone::V12_V13_V14_V15_V16, Capstone::V13_V14_V15_V16_V17,
    Capstone::V14_V15_V16_V17_V18, Capstone::V15_V16_V17_V18_V19,
    Capstone::V16_V17_V18_V19_V20, Capstone::V17_V18_V19_V20_V21,
    Capstone::V18_V19_V20_V21_V22, Capstone::V19_V20_V21_V22_V23};
static const MCPhysReg ArgVRN6M1s[] = {
    Capstone::V8_V9_V10_V11_V12_V13,   Capstone::V9_V10_V11_V12_V13_V14,
    Capstone::V10_V11_V12_V13_V14_V15, Capstone::V11_V12_V13_V14_V15_V16,
    Capstone::V12_V13_V14_V15_V16_V17, Capstone::V13_V14_V15_V16_V17_V18,
    Capstone::V14_V15_V16_V17_V18_V19, Capstone::V15_V16_V17_V18_V19_V20,
    Capstone::V16_V17_V18_V19_V20_V21, Capstone::V17_V18_V19_V20_V21_V22,
    Capstone::V18_V19_V20_V21_V22_V23};
static const MCPhysReg ArgVRN7M1s[] = {
    Capstone::V8_V9_V10_V11_V12_V13_V14,   Capstone::V9_V10_V11_V12_V13_V14_V15,
    Capstone::V10_V11_V12_V13_V14_V15_V16, Capstone::V11_V12_V13_V14_V15_V16_V17,
    Capstone::V12_V13_V14_V15_V16_V17_V18, Capstone::V13_V14_V15_V16_V17_V18_V19,
    Capstone::V14_V15_V16_V17_V18_V19_V20, Capstone::V15_V16_V17_V18_V19_V20_V21,
    Capstone::V16_V17_V18_V19_V20_V21_V22, Capstone::V17_V18_V19_V20_V21_V22_V23};
static const MCPhysReg ArgVRN8M1s[] = {Capstone::V8_V9_V10_V11_V12_V13_V14_V15,
                                       Capstone::V9_V10_V11_V12_V13_V14_V15_V16,
                                       Capstone::V10_V11_V12_V13_V14_V15_V16_V17,
                                       Capstone::V11_V12_V13_V14_V15_V16_V17_V18,
                                       Capstone::V12_V13_V14_V15_V16_V17_V18_V19,
                                       Capstone::V13_V14_V15_V16_V17_V18_V19_V20,
                                       Capstone::V14_V15_V16_V17_V18_V19_V20_V21,
                                       Capstone::V15_V16_V17_V18_V19_V20_V21_V22,
                                       Capstone::V16_V17_V18_V19_V20_V21_V22_V23};
static const MCPhysReg ArgVRN2M2s[] = {Capstone::V8M2_V10M2,  Capstone::V10M2_V12M2,
                                       Capstone::V12M2_V14M2, Capstone::V14M2_V16M2,
                                       Capstone::V16M2_V18M2, Capstone::V18M2_V20M2,
                                       Capstone::V20M2_V22M2};
static const MCPhysReg ArgVRN3M2s[] = {
    Capstone::V8M2_V10M2_V12M2,  Capstone::V10M2_V12M2_V14M2,
    Capstone::V12M2_V14M2_V16M2, Capstone::V14M2_V16M2_V18M2,
    Capstone::V16M2_V18M2_V20M2, Capstone::V18M2_V20M2_V22M2};
static const MCPhysReg ArgVRN4M2s[] = {
    Capstone::V8M2_V10M2_V12M2_V14M2, Capstone::V10M2_V12M2_V14M2_V16M2,
    Capstone::V12M2_V14M2_V16M2_V18M2, Capstone::V14M2_V16M2_V18M2_V20M2,
    Capstone::V16M2_V18M2_V20M2_V22M2};
static const MCPhysReg ArgVRN2M4s[] = {Capstone::V8M4_V12M4, Capstone::V12M4_V16M4,
                                       Capstone::V16M4_V20M4};

ArrayRef<MCPhysReg> Capstone::getArgGPRs(const CapstoneABI::ABI ABI) {
  // The GPRs used for passing arguments in the ILP32* and LP64* ABIs, except
  // the ILP32E ABI.
  static const MCPhysReg ArgIGPRs[] = {Capstone::X10, Capstone::X11, Capstone::X12,
                                       Capstone::X13, Capstone::X14, Capstone::X15,
                                       Capstone::X16, Capstone::X17};
  // The GPRs used for passing arguments in the ILP32E/LP64E ABI.
  static const MCPhysReg ArgEGPRs[] = {Capstone::X10, Capstone::X11, Capstone::X12,
                                       Capstone::X13, Capstone::X14, Capstone::X15};

  if (ABI == CapstoneABI::ABI_ILP32E || ABI == CapstoneABI::ABI_LP64E)
    return ArrayRef(ArgEGPRs);

  return ArrayRef(ArgIGPRs);
}

static ArrayRef<MCPhysReg> getArgGPR16s(const CapstoneABI::ABI ABI) {
  // The GPRs used for passing arguments in the ILP32* and LP64* ABIs, except
  // the ILP32E ABI.
  static const MCPhysReg ArgIGPRs[] = {Capstone::X10_H, Capstone::X11_H, Capstone::X12_H,
                                       Capstone::X13_H, Capstone::X14_H, Capstone::X15_H,
                                       Capstone::X16_H, Capstone::X17_H};
  // The GPRs used for passing arguments in the ILP32E/LP64E ABI.
  static const MCPhysReg ArgEGPRs[] = {Capstone::X10_H, Capstone::X11_H,
                                       Capstone::X12_H, Capstone::X13_H,
                                       Capstone::X14_H, Capstone::X15_H};

  if (ABI == CapstoneABI::ABI_ILP32E || ABI == CapstoneABI::ABI_LP64E)
    return ArrayRef(ArgEGPRs);

  return ArrayRef(ArgIGPRs);
}

static ArrayRef<MCPhysReg> getArgGPR32s(const CapstoneABI::ABI ABI) {
  // The GPRs used for passing arguments in the ILP32* and LP64* ABIs, except
  // the ILP32E ABI.
  static const MCPhysReg ArgIGPRs[] = {Capstone::X10_W, Capstone::X11_W, Capstone::X12_W,
                                       Capstone::X13_W, Capstone::X14_W, Capstone::X15_W,
                                       Capstone::X16_W, Capstone::X17_W};
  // The GPRs used for passing arguments in the ILP32E/LP64E ABI.
  static const MCPhysReg ArgEGPRs[] = {Capstone::X10_W, Capstone::X11_W,
                                       Capstone::X12_W, Capstone::X13_W,
                                       Capstone::X14_W, Capstone::X15_W};

  if (ABI == CapstoneABI::ABI_ILP32E || ABI == CapstoneABI::ABI_LP64E)
    return ArrayRef(ArgEGPRs);

  return ArrayRef(ArgIGPRs);
}

static ArrayRef<MCPhysReg> getFastCCArgGPRs(const CapstoneABI::ABI ABI) {
  // The GPRs used for passing arguments in the FastCC, X5 and X6 might be used
  // for save-restore libcall, so we don't use them.
  // Don't use X7 for fastcc, since Zicfilp uses X7 as the label register.
  static const MCPhysReg FastCCIGPRs[] = {
      Capstone::X10, Capstone::X11, Capstone::X12, Capstone::X13, Capstone::X14, Capstone::X15,
      Capstone::X16, Capstone::X17, Capstone::X28, Capstone::X29, Capstone::X30, Capstone::X31};

  // The GPRs used for passing arguments in the FastCC when using ILP32E/LP64E.
  static const MCPhysReg FastCCEGPRs[] = {Capstone::X10, Capstone::X11, Capstone::X12,
                                          Capstone::X13, Capstone::X14, Capstone::X15};

  if (ABI == CapstoneABI::ABI_ILP32E || ABI == CapstoneABI::ABI_LP64E)
    return ArrayRef(FastCCEGPRs);

  return ArrayRef(FastCCIGPRs);
}

static ArrayRef<MCPhysReg> getFastCCArgGPRF16s(const CapstoneABI::ABI ABI) {
  // The GPRs used for passing arguments in the FastCC, X5 and X6 might be used
  // for save-restore libcall, so we don't use them.
  // Don't use X7 for fastcc, since Zicfilp uses X7 as the label register.
  static const MCPhysReg FastCCIGPRs[] = {
      Capstone::X10_H, Capstone::X11_H, Capstone::X12_H, Capstone::X13_H,
      Capstone::X14_H, Capstone::X15_H, Capstone::X16_H, Capstone::X17_H,
      Capstone::X28_H, Capstone::X29_H, Capstone::X30_H, Capstone::X31_H};

  // The GPRs used for passing arguments in the FastCC when using ILP32E/LP64E.
  static const MCPhysReg FastCCEGPRs[] = {Capstone::X10_H, Capstone::X11_H,
                                          Capstone::X12_H, Capstone::X13_H,
                                          Capstone::X14_H, Capstone::X15_H};

  if (ABI == CapstoneABI::ABI_ILP32E || ABI == CapstoneABI::ABI_LP64E)
    return ArrayRef(FastCCEGPRs);

  return ArrayRef(FastCCIGPRs);
}

static ArrayRef<MCPhysReg> getFastCCArgGPRF32s(const CapstoneABI::ABI ABI) {
  // The GPRs used for passing arguments in the FastCC, X5 and X6 might be used
  // for save-restore libcall, so we don't use them.
  // Don't use X7 for fastcc, since Zicfilp uses X7 as the label register.
  static const MCPhysReg FastCCIGPRs[] = {
      Capstone::X10_W, Capstone::X11_W, Capstone::X12_W, Capstone::X13_W,
      Capstone::X14_W, Capstone::X15_W, Capstone::X16_W, Capstone::X17_W,
      Capstone::X28_W, Capstone::X29_W, Capstone::X30_W, Capstone::X31_W};

  // The GPRs used for passing arguments in the FastCC when using ILP32E/LP64E.
  static const MCPhysReg FastCCEGPRs[] = {Capstone::X10_W, Capstone::X11_W,
                                          Capstone::X12_W, Capstone::X13_W,
                                          Capstone::X14_W, Capstone::X15_W};

  if (ABI == CapstoneABI::ABI_ILP32E || ABI == CapstoneABI::ABI_LP64E)
    return ArrayRef(FastCCEGPRs);

  return ArrayRef(FastCCIGPRs);
}

// Pass a 2*XLEN argument that has been split into two XLEN values through
// registers or the stack as necessary.
static bool CC_CapstoneAssign2XLen(unsigned XLen, CCState &State, CCValAssign VA1,
                                ISD::ArgFlagsTy ArgFlags1, unsigned ValNo2,
                                MVT ValVT2, MVT LocVT2,
                                ISD::ArgFlagsTy ArgFlags2, bool EABI) {
  unsigned XLenInBytes = XLen / 8;
  const CapstoneSubtarget &STI =
      State.getMachineFunction().getSubtarget<CapstoneSubtarget>();
  ArrayRef<MCPhysReg> ArgGPRs = Capstone::getArgGPRs(STI.getTargetABI());

  if (MCRegister Reg = State.AllocateReg(ArgGPRs)) {
    // At least one half can be passed via register.
    State.addLoc(CCValAssign::getReg(VA1.getValNo(), VA1.getValVT(), Reg,
                                     VA1.getLocVT(), CCValAssign::Full));
  } else {
    // Both halves must be passed on the stack, with proper alignment.
    // TODO: To be compatible with GCC's behaviors, we force them to have 4-byte
    // alignment. This behavior may be changed when RV32E/ILP32E is ratified.
    Align StackAlign(XLenInBytes);
    if (!EABI || XLen != 32)
      StackAlign = std::max(StackAlign, ArgFlags1.getNonZeroOrigAlign());
    State.addLoc(
        CCValAssign::getMem(VA1.getValNo(), VA1.getValVT(),
                            State.AllocateStack(XLenInBytes, StackAlign),
                            VA1.getLocVT(), CCValAssign::Full));
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
    return false;
  }

  if (MCRegister Reg = State.AllocateReg(ArgGPRs)) {
    // The second half can also be passed via register.
    State.addLoc(
        CCValAssign::getReg(ValNo2, ValVT2, Reg, LocVT2, CCValAssign::Full));
  } else {
    // The second half is passed via the stack, without additional alignment.
    State.addLoc(CCValAssign::getMem(
        ValNo2, ValVT2, State.AllocateStack(XLenInBytes, Align(XLenInBytes)),
        LocVT2, CCValAssign::Full));
  }

  return false;
}

static MCRegister allocateRVVReg(MVT ValVT, unsigned ValNo, CCState &State,
                                 const CapstoneTargetLowering &TLI) {
  const TargetRegisterClass *RC = TLI.getRegClassFor(ValVT);
  if (RC == &Capstone::VRRegClass) {
    // Assign the first mask argument to V0.
    // This is an interim calling convention and it may be changed in the
    // future.
    if (ValVT.getVectorElementType() == MVT::i1)
      if (MCRegister Reg = State.AllocateReg(Capstone::V0))
        return Reg;
    return State.AllocateReg(ArgVRs);
  }
  if (RC == &Capstone::VRM2RegClass)
    return State.AllocateReg(ArgVRM2s);
  if (RC == &Capstone::VRM4RegClass)
    return State.AllocateReg(ArgVRM4s);
  if (RC == &Capstone::VRM8RegClass)
    return State.AllocateReg(ArgVRM8s);
  if (RC == &Capstone::VRN2M1RegClass)
    return State.AllocateReg(ArgVRN2M1s);
  if (RC == &Capstone::VRN3M1RegClass)
    return State.AllocateReg(ArgVRN3M1s);
  if (RC == &Capstone::VRN4M1RegClass)
    return State.AllocateReg(ArgVRN4M1s);
  if (RC == &Capstone::VRN5M1RegClass)
    return State.AllocateReg(ArgVRN5M1s);
  if (RC == &Capstone::VRN6M1RegClass)
    return State.AllocateReg(ArgVRN6M1s);
  if (RC == &Capstone::VRN7M1RegClass)
    return State.AllocateReg(ArgVRN7M1s);
  if (RC == &Capstone::VRN8M1RegClass)
    return State.AllocateReg(ArgVRN8M1s);
  if (RC == &Capstone::VRN2M2RegClass)
    return State.AllocateReg(ArgVRN2M2s);
  if (RC == &Capstone::VRN3M2RegClass)
    return State.AllocateReg(ArgVRN3M2s);
  if (RC == &Capstone::VRN4M2RegClass)
    return State.AllocateReg(ArgVRN4M2s);
  if (RC == &Capstone::VRN2M4RegClass)
    return State.AllocateReg(ArgVRN2M4s);
  llvm_unreachable("Unhandled register class for ValueType");
}

// Implements the Capstone calling convention. Returns true upon failure.
bool llvm::CC_Capstone(unsigned ValNo, MVT ValVT, MVT LocVT,
                    CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                    CCState &State, bool IsRet, Type *OrigTy) {
  const MachineFunction &MF = State.getMachineFunction();
  const DataLayout &DL = MF.getDataLayout();
  const CapstoneSubtarget &Subtarget = MF.getSubtarget<CapstoneSubtarget>();
  const CapstoneTargetLowering &TLI = *Subtarget.getTargetLowering();

  unsigned XLen = Subtarget.getXLen();
  MVT XLenVT = Subtarget.getXLenVT();

  if (ArgFlags.isNest()) {
    // Static chain parameter must not be passed in normal argument registers,
    // so we assign t2/t3 for it as done in GCC's
    // __builtin_call_with_static_chain
    bool HasCFBranch =
        Subtarget.hasStdExtZicfilp() &&
        MF.getFunction().getParent()->getModuleFlag("cf-protection-branch");

    // Normal: t2, Branch control flow protection: t3
    const auto StaticChainReg = HasCFBranch ? Capstone::X28 : Capstone::X7;

    CapstoneABI::ABI ABI = Subtarget.getTargetABI();
    if (HasCFBranch &&
        (ABI == CapstoneABI::ABI_ILP32E || ABI == CapstoneABI::ABI_LP64E))
      reportFatalUsageError(
          "Nested functions with control flow protection are not "
          "usable with ILP32E or LP64E ABI.");
    if (MCRegister Reg = State.AllocateReg(StaticChainReg)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  // Any return value split in to more than two values can't be returned
  // directly. Vectors are returned via the available vector registers.
  if (!LocVT.isVector() && IsRet && ValNo > 1)
    return true;

  // UseGPRForF16_F32 if targeting one of the soft-float ABIs, if passing a
  // variadic argument, or if no F16/F32 argument registers are available.
  bool UseGPRForF16_F32 = true;
  // UseGPRForF64 if targeting soft-float ABIs or an FLEN=32 ABI, if passing a
  // variadic argument, or if no F64 argument registers are available.
  bool UseGPRForF64 = true;

  CapstoneABI::ABI ABI = Subtarget.getTargetABI();
  switch (ABI) {
  default:
    llvm_unreachable("Unexpected ABI");
  case CapstoneABI::ABI_ILP32:
  case CapstoneABI::ABI_ILP32E:
  case CapstoneABI::ABI_LP64:
  case CapstoneABI::ABI_LP64E:
    break;
  case CapstoneABI::ABI_ILP32F:
  case CapstoneABI::ABI_LP64F:
    UseGPRForF16_F32 = ArgFlags.isVarArg();
    break;
  case CapstoneABI::ABI_ILP32D:
  case CapstoneABI::ABI_LP64D:
    UseGPRForF16_F32 = ArgFlags.isVarArg();
    UseGPRForF64 = ArgFlags.isVarArg();
    break;
  }

  if ((LocVT == MVT::f16 || LocVT == MVT::bf16) && !UseGPRForF16_F32) {
    if (MCRegister Reg = State.AllocateReg(ArgFPR16s)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32 && !UseGPRForF16_F32) {
    if (MCRegister Reg = State.AllocateReg(ArgFPR32s)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64 && !UseGPRForF64) {
    if (MCRegister Reg = State.AllocateReg(ArgFPR64s)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if ((ValVT == MVT::f16 && Subtarget.hasStdExtZhinxmin())) {
    if (MCRegister Reg = State.AllocateReg(getArgGPR16s(ABI))) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (ValVT == MVT::f32 && Subtarget.hasStdExtZfinx()) {
    if (MCRegister Reg = State.AllocateReg(getArgGPR32s(ABI))) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  ArrayRef<MCPhysReg> ArgGPRs = Capstone::getArgGPRs(ABI);

  // Zdinx use GPR without a bitcast when possible.
  if (LocVT == MVT::f64 && XLen == 64 && Subtarget.hasStdExtZdinx()) {
    if (MCRegister Reg = State.AllocateReg(ArgGPRs)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  // FP smaller than XLen, uses custom GPR.
  if (LocVT == MVT::f16 || LocVT == MVT::bf16 ||
      (LocVT == MVT::f32 && XLen == 64)) {
    if (MCRegister Reg = State.AllocateReg(ArgGPRs)) {
      LocVT = XLenVT;
      State.addLoc(
          CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  // Bitcast FP to GPR if we can use a GPR register.
  if ((XLen == 32 && LocVT == MVT::f32) || (XLen == 64 && LocVT == MVT::f64)) {
    if (MCRegister Reg = State.AllocateReg(ArgGPRs)) {
      LocVT = XLenVT;
      LocInfo = CCValAssign::BCvt;
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  // If this is a variadic argument, the Capstone calling convention requires
  // that it is assigned an 'even' or 'aligned' register if it has 8-byte
  // alignment (RV32) or 16-byte alignment (RV64). An aligned register should
  // be used regardless of whether the original argument was split during
  // legalisation or not. The argument will not be passed by registers if the
  // original type is larger than 2*XLEN, so the register alignment rule does
  // not apply.
  // TODO: To be compatible with GCC's behaviors, we don't align registers
  // currently if we are using ILP32E calling convention. This behavior may be
  // changed when RV32E/ILP32E is ratified.
  unsigned TwoXLenInBytes = (2 * XLen) / 8;
  if (ArgFlags.isVarArg() && ArgFlags.getNonZeroOrigAlign() == TwoXLenInBytes &&
      DL.getTypeAllocSize(OrigTy) == TwoXLenInBytes &&
      ABI != CapstoneABI::ABI_ILP32E) {
    unsigned RegIdx = State.getFirstUnallocated(ArgGPRs);
    // Skip 'odd' register if necessary.
    if (RegIdx != std::size(ArgGPRs) && RegIdx % 2 == 1)
      State.AllocateReg(ArgGPRs);
  }

  SmallVectorImpl<CCValAssign> &PendingLocs = State.getPendingLocs();
  SmallVectorImpl<ISD::ArgFlagsTy> &PendingArgFlags =
      State.getPendingArgFlags();

  assert(PendingLocs.size() == PendingArgFlags.size() &&
         "PendingLocs and PendingArgFlags out of sync");

  // Handle passing f64 on RV32D with a soft float ABI or when floating point
  // registers are exhausted.
  if (XLen == 32 && LocVT == MVT::f64) {
    assert(PendingLocs.empty() && "Can't lower f64 if it is split");
    // Depending on available argument GPRS, f64 may be passed in a pair of
    // GPRs, split between a GPR and the stack, or passed completely on the
    // stack. LowerCall/LowerFormalArguments/LowerReturn must recognise these
    // cases.
    MCRegister Reg = State.AllocateReg(ArgGPRs);
    if (!Reg) {
      int64_t StackOffset = State.AllocateStack(8, Align(8));
      State.addLoc(
          CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
      return false;
    }
    LocVT = MVT::i32;
    State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    MCRegister HiReg = State.AllocateReg(ArgGPRs);
    if (HiReg) {
      State.addLoc(
          CCValAssign::getCustomReg(ValNo, ValVT, HiReg, LocVT, LocInfo));
    } else {
      int64_t StackOffset = State.AllocateStack(4, Align(4));
      State.addLoc(
          CCValAssign::getCustomMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
    }
    return false;
  }

  // Split arguments might be passed indirectly, so keep track of the pending
  // values. Split vectors are passed via a mix of registers and indirectly, so
  // treat them as we would any other argument.
  if (ValVT.isScalarInteger() && (ArgFlags.isSplit() || !PendingLocs.empty())) {
    LocVT = XLenVT;
    LocInfo = CCValAssign::Indirect;
    PendingLocs.push_back(
        CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));
    PendingArgFlags.push_back(ArgFlags);
    if (!ArgFlags.isSplitEnd()) {
      return false;
    }
  }

  // If the split argument only had two elements, it should be passed directly
  // in registers or on the stack.
  if (ValVT.isScalarInteger() && ArgFlags.isSplitEnd() &&
      PendingLocs.size() <= 2) {
    assert(PendingLocs.size() == 2 && "Unexpected PendingLocs.size()");
    // Apply the normal calling convention rules to the first half of the
    // split argument.
    CCValAssign VA = PendingLocs[0];
    ISD::ArgFlagsTy AF = PendingArgFlags[0];
    PendingLocs.clear();
    PendingArgFlags.clear();
    return CC_CapstoneAssign2XLen(
        XLen, State, VA, AF, ValNo, ValVT, LocVT, ArgFlags,
        ABI == CapstoneABI::ABI_ILP32E || ABI == CapstoneABI::ABI_LP64E);
  }

  // Allocate to a register if possible, or else a stack slot.
  MCRegister Reg;
  unsigned StoreSizeBytes = XLen / 8;
  Align StackAlign = Align(XLen / 8);

  if (ValVT.isVector() || ValVT.isCapstoneVectorTuple()) {
    Reg = allocateRVVReg(ValVT, ValNo, State, TLI);
    if (Reg) {
      // Fixed-length vectors are located in the corresponding scalable-vector
      // container types.
      if (ValVT.isFixedLengthVector()) {
        LocVT = TLI.getContainerForFixedLengthVector(LocVT);
        State.addLoc(
            CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
        return false;
      }
    } else {
      // For return values, the vector must be passed fully via registers or
      // via the stack.
      // FIXME: The proposed vector ABI only mandates v8-v15 for return values,
      // but we're using all of them.
      if (IsRet)
        return true;
      // Try using a GPR to pass the address
      if ((Reg = State.AllocateReg(ArgGPRs))) {
        LocVT = XLenVT;
        LocInfo = CCValAssign::Indirect;
      } else if (ValVT.isScalableVector()) {
        LocVT = XLenVT;
        LocInfo = CCValAssign::Indirect;
      } else {
        StoreSizeBytes = ValVT.getStoreSize();
        // Align vectors to their element sizes, being careful for vXi1
        // vectors.
        StackAlign = MaybeAlign(ValVT.getScalarSizeInBits() / 8).valueOrOne();
      }
    }
  } else {
    Reg = State.AllocateReg(ArgGPRs);
  }

  int64_t StackOffset =
      Reg ? 0 : State.AllocateStack(StoreSizeBytes, StackAlign);

  // If we reach this point and PendingLocs is non-empty, we must be at the
  // end of a split argument that must be passed indirectly.
  if (!PendingLocs.empty()) {
    assert(ArgFlags.isSplitEnd() && "Expected ArgFlags.isSplitEnd()");
    assert(PendingLocs.size() > 2 && "Unexpected PendingLocs.size()");

    for (auto &It : PendingLocs) {
      if (Reg)
        It.convertToReg(Reg);
      else
        It.convertToMem(StackOffset);
      State.addLoc(It);
    }
    PendingLocs.clear();
    PendingArgFlags.clear();
    return false;
  }

  assert(((ValVT.isFloatingPoint() && !ValVT.isVector()) || LocVT == XLenVT ||
          (TLI.getSubtarget().hasVInstructions() &&
           (ValVT.isVector() || ValVT.isCapstoneVectorTuple()))) &&
         "Expected an XLenVT or vector types at this stage");

  if (Reg) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  State.addLoc(CCValAssign::getMem(ValNo, ValVT, StackOffset, LocVT, LocInfo));
  return false;
}

// FastCC has less than 1% performance improvement for some particular
// benchmark. But theoretically, it may have benefit for some cases.
bool llvm::CC_Capstone_FastCC(unsigned ValNo, MVT ValVT, MVT LocVT,
                           CCValAssign::LocInfo LocInfo,
                           ISD::ArgFlagsTy ArgFlags, CCState &State, bool IsRet,
                           Type *OrigTy) {
  const MachineFunction &MF = State.getMachineFunction();
  const CapstoneSubtarget &Subtarget = MF.getSubtarget<CapstoneSubtarget>();
  const CapstoneTargetLowering &TLI = *Subtarget.getTargetLowering();
  CapstoneABI::ABI ABI = Subtarget.getTargetABI();

  if ((LocVT == MVT::f16 && Subtarget.hasStdExtZfhmin()) ||
      (LocVT == MVT::bf16 && Subtarget.hasStdExtZfbfmin())) {
    static const MCPhysReg FPR16List[] = {
        Capstone::F10_H, Capstone::F11_H, Capstone::F12_H, Capstone::F13_H, Capstone::F14_H,
        Capstone::F15_H, Capstone::F16_H, Capstone::F17_H, Capstone::F0_H,  Capstone::F1_H,
        Capstone::F2_H,  Capstone::F3_H,  Capstone::F4_H,  Capstone::F5_H,  Capstone::F6_H,
        Capstone::F7_H,  Capstone::F28_H, Capstone::F29_H, Capstone::F30_H, Capstone::F31_H};
    if (MCRegister Reg = State.AllocateReg(FPR16List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32 && Subtarget.hasStdExtF()) {
    static const MCPhysReg FPR32List[] = {
        Capstone::F10_F, Capstone::F11_F, Capstone::F12_F, Capstone::F13_F, Capstone::F14_F,
        Capstone::F15_F, Capstone::F16_F, Capstone::F17_F, Capstone::F0_F,  Capstone::F1_F,
        Capstone::F2_F,  Capstone::F3_F,  Capstone::F4_F,  Capstone::F5_F,  Capstone::F6_F,
        Capstone::F7_F,  Capstone::F28_F, Capstone::F29_F, Capstone::F30_F, Capstone::F31_F};
    if (MCRegister Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64 && Subtarget.hasStdExtD()) {
    static const MCPhysReg FPR64List[] = {
        Capstone::F10_D, Capstone::F11_D, Capstone::F12_D, Capstone::F13_D, Capstone::F14_D,
        Capstone::F15_D, Capstone::F16_D, Capstone::F17_D, Capstone::F0_D,  Capstone::F1_D,
        Capstone::F2_D,  Capstone::F3_D,  Capstone::F4_D,  Capstone::F5_D,  Capstone::F6_D,
        Capstone::F7_D,  Capstone::F28_D, Capstone::F29_D, Capstone::F30_D, Capstone::F31_D};
    if (MCRegister Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  MVT XLenVT = Subtarget.getXLenVT();

  // Check if there is an available GPRF16 before hitting the stack.
  if ((LocVT == MVT::f16 && Subtarget.hasStdExtZhinxmin())) {
    if (MCRegister Reg = State.AllocateReg(getFastCCArgGPRF16s(ABI))) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  // Check if there is an available GPRF32 before hitting the stack.
  if (LocVT == MVT::f32 && Subtarget.hasStdExtZfinx()) {
    if (MCRegister Reg = State.AllocateReg(getFastCCArgGPRF32s(ABI))) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  // Check if there is an available GPR before hitting the stack.
  if (LocVT == MVT::f64 && Subtarget.is64Bit() && Subtarget.hasStdExtZdinx()) {
    if (MCRegister Reg = State.AllocateReg(getFastCCArgGPRs(ABI))) {
      if (LocVT.getSizeInBits() != Subtarget.getXLen()) {
        LocVT = XLenVT;
        State.addLoc(
            CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
        return false;
      }
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  ArrayRef<MCPhysReg> ArgGPRs = getFastCCArgGPRs(ABI);

  if (LocVT.isVector()) {
    if (MCRegister Reg = allocateRVVReg(ValVT, ValNo, State, TLI)) {
      // Fixed-length vectors are located in the corresponding scalable-vector
      // container types.
      if (LocVT.isFixedLengthVector()) {
        LocVT = TLI.getContainerForFixedLengthVector(LocVT);
        State.addLoc(
            CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
        return false;
      }
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }

    // Pass scalable vectors indirectly. Pass fixed vectors indirectly if we
    // have a free GPR.
    if (LocVT.isScalableVector() ||
        State.getFirstUnallocated(ArgGPRs) != ArgGPRs.size()) {
      LocInfo = CCValAssign::Indirect;
      LocVT = XLenVT;
    }
  }

  if (LocVT == XLenVT) {
    if (MCRegister Reg = State.AllocateReg(getFastCCArgGPRs(ABI))) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == XLenVT || LocVT == MVT::f16 || LocVT == MVT::bf16 ||
      LocVT == MVT::f32 || LocVT == MVT::f64 || LocVT.isFixedLengthVector()) {
    Align StackAlign = MaybeAlign(ValVT.getScalarSizeInBits() / 8).valueOrOne();
    int64_t Offset = State.AllocateStack(LocVT.getStoreSize(), StackAlign);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
    return false;
  }

  return true; // CC didn't match.
}

bool llvm::CC_Capstone_GHC(unsigned ValNo, MVT ValVT, MVT LocVT,
                        CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                        Type *OrigTy, CCState &State) {
  if (ArgFlags.isNest()) {
    report_fatal_error(
        "Attribute 'nest' is not supported in GHC calling convention");
  }

  static const MCPhysReg GPRList[] = {
      Capstone::X9,  Capstone::X18, Capstone::X19, Capstone::X20, Capstone::X21, Capstone::X22,
      Capstone::X23, Capstone::X24, Capstone::X25, Capstone::X26, Capstone::X27};

  if (LocVT == MVT::i32 || LocVT == MVT::i64) {
    // Pass in STG registers: Base, Sp, Hp, R1, R2, R3, R4, R5, R6, R7, SpLim
    //                        s1    s2  s3  s4  s5  s6  s7  s8  s9  s10 s11
    if (MCRegister Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  const CapstoneSubtarget &Subtarget =
      State.getMachineFunction().getSubtarget<CapstoneSubtarget>();

  if (LocVT == MVT::f32 && Subtarget.hasStdExtF()) {
    // Pass in STG registers: F1, ..., F6
    //                        fs0 ... fs5
    static const MCPhysReg FPR32List[] = {Capstone::F8_F,  Capstone::F9_F,
                                          Capstone::F18_F, Capstone::F19_F,
                                          Capstone::F20_F, Capstone::F21_F};
    if (MCRegister Reg = State.AllocateReg(FPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64 && Subtarget.hasStdExtD()) {
    // Pass in STG registers: D1, ..., D6
    //                        fs6 ... fs11
    static const MCPhysReg FPR64List[] = {Capstone::F22_D, Capstone::F23_D,
                                          Capstone::F24_D, Capstone::F25_D,
                                          Capstone::F26_D, Capstone::F27_D};
    if (MCRegister Reg = State.AllocateReg(FPR64List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f32 && Subtarget.hasStdExtZfinx()) {
    static const MCPhysReg GPR32List[] = {
        Capstone::X9_W,  Capstone::X18_W, Capstone::X19_W, Capstone::X20_W,
        Capstone::X21_W, Capstone::X22_W, Capstone::X23_W, Capstone::X24_W,
        Capstone::X25_W, Capstone::X26_W, Capstone::X27_W};
    if (MCRegister Reg = State.AllocateReg(GPR32List)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::f64 && Subtarget.hasStdExtZdinx() && Subtarget.is64Bit()) {
    if (MCRegister Reg = State.AllocateReg(GPRList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  report_fatal_error("No registers left in GHC calling convention");
  return true;
}
