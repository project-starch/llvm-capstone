//===-- CapstoneTargetMachine.h - Define TargetMachine for Capstone --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Capstone specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_CapstoneTARGETMACHINE_H
#define LLVM_LIB_TARGET_Capstone_CapstoneTARGETMACHINE_H

#include "MCTargetDesc/CapstoneMCTargetDesc.h"
#include "CapstoneSubtarget.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/IR/DataLayout.h"
#include <optional>

namespace llvm {
class CapstoneTargetMachine : public CodeGenTargetMachineImpl {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  mutable StringMap<std::unique_ptr<CapstoneSubtarget>> SubtargetMap;

public:
  CapstoneTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     std::optional<Reloc::Model> RM,
                     std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                     bool JIT);

  const CapstoneSubtarget *getSubtargetImpl(const Function &F) const override;
  // DO NOT IMPLEMENT: There is no such thing as a valid default subtarget,
  // subtargets are per-function entities based on the target-specific
  // attributes of each function.
  const CapstoneSubtarget *getSubtargetImpl() const = delete;

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  MachineFunctionInfo *
  createMachineFunctionInfo(BumpPtrAllocator &Allocator, const Function &F,
                            const TargetSubtargetInfo *STI) const override;

  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;

  bool isNoopAddrSpaceCast(unsigned SrcAS, unsigned DstAS) const override;

  /// The address space of memory reached through a pseudo-source value. Stack
  /// slots, constant pools, jump tables and the GOT are all reached through
  /// capabilities (addrspace 200), and a MachinePointerInfo built from such a
  /// pseudo source must say so: the DAG's CSE key for a load or store includes
  /// the pointer info's address space, and a load whose pointer folds to a frame
  /// index gets its pointer info re-inferred from that frame index. With the
  /// default (0) the re-inferred info differed from the original (200, from the
  /// unknown-stack info) and DAGCombiner's alignment refinement, which expects
  /// to get the SAME node back from getExtLoad, got a fresh one instead
  /// (F-02/F-03: `NewLoad.getNode() == N` asserted on every variable-index
  /// extractelement/insertelement at -O2).
  unsigned getAddressSpaceForPseudoSourceKind(unsigned Kind) const override;

  yaml::MachineFunctionInfo *createDefaultFuncInfoYAML() const override;
  yaml::MachineFunctionInfo *
  convertFuncInfoToYAML(const MachineFunction &MF) const override;
  bool parseMachineFunctionInfo(const yaml::MachineFunctionInfo &,
                                PerFunctionMIParsingState &PFS,
                                SMDiagnostic &Error,
                                SMRange &SourceRange) const override;
  void registerPassBuilderCallbacks(PassBuilder &PB) override;
  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override;
  ScheduleDAGInstrs *
  createPostMachineScheduler(MachineSchedContext *C) const override;
};

std::unique_ptr<ScheduleDAGMutation>
createCapstoneVectorMaskDAGMutation(const TargetRegisterInfo *TRI);

} // namespace llvm

#endif
