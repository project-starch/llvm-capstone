//===-- CapstoneTargetMachine.cpp - Define TargetMachine for Capstone -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Capstone target spec.
//
//===----------------------------------------------------------------------===//

#include "CapstoneTargetMachine.h"
#include "Capstone.h"
#include "CapstoneTargetObjectFile.h"
#include "CapstoneTargetTransformInfo.h"
#include "TargetInfo/CapstoneTargetInfo.h" // For getTheCapstoneTarget.
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h" // For RegisterTargetMachine.
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"  // For CodeGenOptLevel.
#include "llvm/Support/Compiler.h" // For LLVM_EXTERNAL_VISIBILITY.
#include <memory>

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeCapstoneTarget() {
  // Register the target so that external tools can instantiate it.
  RegisterTargetMachine<CapstoneTargetMachine> X(getTheCapstoneTarget());

  // PassRegistry &PR = *PassRegistry::getPassRegistry();
  // initializeCapstoneSimpleConstantPropagationPass(PR);
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  if (TT.isOSBinFormatELF())
    return std::make_unique<Capstone_ELFTargetObjectFile>();
  if (TT.isOSBinFormatMachO())
    return std::make_unique<Capstone_MachoTargetObjectFile>();
  // Other format not supported yet.
  return nullptr;
}

// TODO: Share this with Clang.
static const char *CapstoneDataLayoutStr =
    "e-p:16:16:16-n16:32-i32:32:32-i16:16:16-i1:8:8-f32:32:32-v32:32:32";

CapstoneTargetMachine::CapstoneTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, CapstoneDataLayoutStr, TT, CPU, FS, Options,
                               // Use the simplest relocation by default.
                               RM ? *RM : Reloc::Static,
                               CM ? *CM : CodeModel::Small, OL),
      TLOF(createTLOF(getTargetTriple())) {
  initAsmInfo();
}

CapstoneTargetMachine::~CapstoneTargetMachine() = default;

const CapstoneSubtarget *
CapstoneTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  StringRef CPU = CPUAttr.isValid() ? CPUAttr.getValueAsString() : TargetCPU;
  StringRef FS = FSAttr.isValid() ? FSAttr.getValueAsString() : TargetFS;

  // Eventually, we'll want to hook up a different subtarget based on at the
  // target feature, target cpu, and tune cpu attached to F, but as of now,
  // the target doesn't support anything fancy so we just have one subtarget
  // for everything.
  if (!SubtargetSingleton)
    SubtargetSingleton =
        std::make_unique<CapstoneSubtarget>(TargetTriple, CPU, FS, *this);
  return SubtargetSingleton.get();
}

TargetTransformInfo
CapstoneTargetMachine::getTargetTransformInfo(const Function &F) const {
  // return TargetTransformInfo(CapstoneTTIImpl(this, F));
  return TargetTransformInfo(std::make_unique<CapstoneTTIImpl>(this, F));
}

void CapstoneTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
// #define GET_PASS_REGISTRY "CapstonePassRegistry.def"
// #include "llvm/Passes/TargetPassRegistry.inc"

  /*
  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel OptLevel) {
        // Do not add optimization passes if we are in O0.
        if (OptLevel == OptimizationLevel::O0)
          return;
        FunctionPassManager FPM;
        FPM.addPass(CapstoneSimpleConstantPropagationNewPass());
        MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      });
      */
}

TargetPassConfig *CapstoneTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new CapstonePassConfig(*this, PM);
}

CapstonePassConfig::CapstonePassConfig(TargetMachine &TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

bool CapstonePassConfig::addInstSelector() {
  // TODO: We need to hook up the DAG selector here.
  return false;
}

void CapstonePassConfig::addIRPasses() {
  /*
  // Add the regular IR passes before putting our passes.
  TargetPassConfig::addIRPasses();
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createCapstoneSimpleConstantPropagationPassForLegacyPM());
    */
}
