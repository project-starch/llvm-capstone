//=-- CapstoneTargetMachine.h - Define TargetMachine for Capstone -------*- C++ -*-=//
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

#include "CapstoneSubtarget.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include <memory>
#include <optional>

namespace llvm {

class CapstoneTargetMachine : public CodeGenTargetMachineImpl {
  mutable std::unique_ptr<CapstoneSubtarget> SubtargetSingleton;
  std::unique_ptr<TargetLoweringObjectFile> TLOF;

public:
  CapstoneTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     std::optional<Reloc::Model> RM,
                     std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                     bool JIT);
  ~CapstoneTargetMachine() override;

  const CapstoneSubtarget *getSubtargetImpl(const Function &F) const override;
  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  // Register the target specific passes that this backend offers.
  void registerPassBuilderCallbacks(PassBuilder &PB) override;
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
};

class CapstonePassConfig : public TargetPassConfig {
public:
  CapstonePassConfig(TargetMachine &TM, PassManagerBase &PM);

  bool addInstSelector() override;
  void addIRPasses() override;
};

} // end namespace llvm

#endif
