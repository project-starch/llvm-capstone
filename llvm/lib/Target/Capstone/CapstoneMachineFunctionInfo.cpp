//=- CapstoneMachineFunctionInfo.cpp - Capstone machine function info --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares Capstone-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#include "CapstoneMachineFunctionInfo.h"
#include "llvm/IR/Module.h"

using namespace llvm;

yaml::CapstoneMachineFunctionInfo::CapstoneMachineFunctionInfo(
    const llvm::CapstoneMachineFunctionInfo &MFI)
    : VarArgsFrameIndex(MFI.getVarArgsFrameIndex()),
      VarArgsSaveSize(MFI.getVarArgsSaveSize()) {}

MachineFunctionInfo *CapstoneMachineFunctionInfo::clone(
    BumpPtrAllocator &Allocator, MachineFunction &DestMF,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
    const {
  return DestMF.cloneInfo<CapstoneMachineFunctionInfo>(*this);
}

CapstoneMachineFunctionInfo::CapstoneMachineFunctionInfo(const Function &F,
                                                   const CapstoneSubtarget *STI) {

  // The default stack probe size is 4096 if the function has no
  // stack-probe-size attribute. This is a safe default because it is the
  // smallest possible guard page size.
  uint64_t ProbeSize = 4096;
  if (F.hasFnAttribute("stack-probe-size"))
    ProbeSize = F.getFnAttributeAsParsedInteger("stack-probe-size");
  else if (const auto *PS = mdconst::extract_or_null<ConstantInt>(
               F.getParent()->getModuleFlag("stack-probe-size")))
    ProbeSize = PS->getZExtValue();
  assert(int64_t(ProbeSize) > 0 && "Invalid stack probe size");

  // Round down to the stack alignment.
  uint64_t StackAlign =
      STI->getFrameLowering()->getTransientStackAlign().value();
  ProbeSize = std::max(StackAlign, alignDown(ProbeSize, StackAlign));
  StringRef ProbeKind;
  if (F.hasFnAttribute("probe-stack"))
    ProbeKind = F.getFnAttribute("probe-stack").getValueAsString();
  else if (const auto *PS = dyn_cast_or_null<MDString>(
               F.getParent()->getModuleFlag("probe-stack")))
    ProbeKind = PS->getString();
  if (ProbeKind.size()) {
    StackProbeSize = ProbeSize;
  }
}

CapstoneMachineFunctionInfo::InterruptStackKind
CapstoneMachineFunctionInfo::getInterruptStackKind(
    const MachineFunction &MF) const {
  if (!MF.getFunction().hasFnAttribute("interrupt"))
    return InterruptStackKind::None;

  assert(VarArgsSaveSize == 0 &&
         "Interrupt functions should not having incoming varargs");

  StringRef InterruptVal =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

  return StringSwitch<CapstoneMachineFunctionInfo::InterruptStackKind>(
             InterruptVal)
      .Case("qci-nest", InterruptStackKind::QCINest)
      .Case("qci-nonest", InterruptStackKind::QCINoNest)
      .Case("SiFive-CLIC-preemptible",
            InterruptStackKind::SiFiveCLICPreemptible)
      .Case("SiFive-CLIC-stack-swap", InterruptStackKind::SiFiveCLICStackSwap)
      .Case("SiFive-CLIC-preemptible-stack-swap",
            InterruptStackKind::SiFiveCLICPreemptibleStackSwap)
      .Default(InterruptStackKind::None);
}

void yaml::CapstoneMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<CapstoneMachineFunctionInfo>::mapping(YamlIO, *this);
}

CapstoneMachineFunctionInfo::PushPopKind
CapstoneMachineFunctionInfo::getPushPopKind(const MachineFunction &MF) const {
  // We cannot use fixed locations for the callee saved spill slots if the
  // function uses a varargs save area.
  // TODO: Use a separate placement for vararg registers to enable Zcmp.
  if (VarArgsSaveSize != 0)
    return PushPopKind::None;

  // SiFive interrupts are not compatible with push/pop.
  if (useSiFiveInterrupt(MF))
    return PushPopKind::None;

  // Zcmp is not compatible with the frame pointer convention.
  if (MF.getSubtarget<CapstoneSubtarget>().hasStdExtZcmp() &&
      !MF.getTarget().Options.DisableFramePointerElim(MF))
    return PushPopKind::StdExtZcmp;

  // Xqccmp is Zcmp but has a push order compatible with the frame-pointer
  // convention.
  if (MF.getSubtarget<CapstoneSubtarget>().hasVendorXqccmp())
    return PushPopKind::VendorXqccmp;

  return PushPopKind::None;
}

bool CapstoneMachineFunctionInfo::hasImplicitFPUpdates(
    const MachineFunction &MF) const {
  switch (getInterruptStackKind(MF)) {
  case InterruptStackKind::QCINest:
  case InterruptStackKind::QCINoNest:
    // QC.C.MIENTER and QC.C.MIENTER.NEST both update FP on function entry.
    return true;
  default:
    break;
  }

  switch (getPushPopKind(MF)) {
  case PushPopKind::VendorXqccmp:
    // When using Xqccmp, we will use `QC.CM.PUSHFP` when Frame Pointers are
    // enabled, which will update FP.
    return true;
  default:
    break;
  }

  return false;
}

void CapstoneMachineFunctionInfo::initializeBaseYamlFields(
    const yaml::CapstoneMachineFunctionInfo &YamlMFI) {
  VarArgsFrameIndex = YamlMFI.VarArgsFrameIndex;
  VarArgsSaveSize = YamlMFI.VarArgsSaveSize;
}

void CapstoneMachineFunctionInfo::addSExt32Register(Register Reg) {
  SExt32Registers.push_back(Reg);
}

bool CapstoneMachineFunctionInfo::isSExt32Register(Register Reg) const {
  return is_contained(SExt32Registers, Reg);
}
