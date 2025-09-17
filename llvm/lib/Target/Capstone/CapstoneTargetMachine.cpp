//===-- CapstoneTargetMachine.cpp - Define TargetMachine for Capstone ----------===//
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
#include "MCTargetDesc/CapstoneBaseInfo.h"
#include "Capstone.h"
#include "CapstoneMachineFunctionInfo.h"
#include "CapstoneTargetObjectFile.h"
#include "CapstoneTargetTransformInfo.h"
#include "TargetInfo/CapstoneTargetInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize/LoopIdiomVectorize.h"
#include <optional>
using namespace llvm;

static cl::opt<bool> EnableRedundantCopyElimination(
    "capstone-enable-copyelim",
    cl::desc("Enable the redundant copy elimination pass"), cl::init(true),
    cl::Hidden);

// FIXME: Unify control over GlobalMerge.
static cl::opt<cl::boolOrDefault>
    EnableGlobalMerge("capstone-enable-global-merge", cl::Hidden,
                      cl::desc("Enable the global merge pass"));

static cl::opt<bool>
    EnableMachineCombiner("capstone-enable-machine-combiner",
                          cl::desc("Enable the machine combiner pass"),
                          cl::init(true), cl::Hidden);

static cl::opt<unsigned> RVVVectorBitsMaxOpt(
    "capstone-v-vector-bits-max",
    cl::desc("Assume V extension vector registers are at most this big, "
             "with zero meaning no maximum size is assumed."),
    cl::init(0), cl::Hidden);

static cl::opt<int> RVVVectorBitsMinOpt(
    "capstone-v-vector-bits-min",
    cl::desc("Assume V extension vector registers are at least this big, "
             "with zero meaning no minimum size is assumed. A value of -1 "
             "means use Zvl*b extension. This is primarily used to enable "
             "autovectorization with fixed width vectors."),
    cl::init(-1), cl::Hidden);

static cl::opt<bool> EnableCapstoneCopyPropagation(
    "capstone-enable-copy-propagation",
    cl::desc("Enable the copy propagation with Capstone copy instr"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> EnableCapstoneDeadRegisterElimination(
    "capstone-enable-dead-defs", cl::Hidden,
    cl::desc("Enable the pass that removes dead"
             " definitions and replaces stores to"
             " them with stores to x0"),
    cl::init(true));

static cl::opt<bool>
    EnableSinkFold("capstone-enable-sink-fold",
                   cl::desc("Enable sinking and folding of instruction copies"),
                   cl::init(true), cl::Hidden);

static cl::opt<bool>
    EnableLoopDataPrefetch("capstone-enable-loop-data-prefetch", cl::Hidden,
                           cl::desc("Enable the loop data prefetch pass"),
                           cl::init(true));

static cl::opt<bool> DisableVectorMaskMutation(
    "capstone-disable-vector-mask-mutation",
    cl::desc("Disable the vector mask scheduling mutation"), cl::init(false),
    cl::Hidden);

static cl::opt<bool>
    EnableMachinePipeliner("capstone-enable-pipeliner",
                           cl::desc("Enable Machine Pipeliner for Capstone"),
                           cl::init(false), cl::Hidden);

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeCapstoneTarget() {
  RegisterTargetMachine<CapstoneTargetMachine> X(getTheCapstone32Target());
  RegisterTargetMachine<CapstoneTargetMachine> Y(getTheCapstone64Target());
  RegisterTargetMachine<CapstoneTargetMachine> A(getTheCapstone32beTarget());
  RegisterTargetMachine<CapstoneTargetMachine> B(getTheCapstone64beTarget());
  auto *PR = PassRegistry::getPassRegistry();
  initializeGlobalISel(*PR);
  initializeCapstoneO0PreLegalizerCombinerPass(*PR);
  initializeCapstonePreLegalizerCombinerPass(*PR);
  initializeCapstonePostLegalizerCombinerPass(*PR);
  initializeKCFIPass(*PR);
  initializeCapstoneDeadRegisterDefinitionsPass(*PR);
  initializeCapstoneLateBranchOptPass(*PR);
  initializeCapstoneMakeCompressibleOptPass(*PR);
  initializeCapstoneGatherScatterLoweringPass(*PR);
  initializeCapstoneCodeGenPreparePass(*PR);
  initializeCapstonePostRAExpandPseudoPass(*PR);
  initializeCapstoneMergeBaseOffsetOptPass(*PR);
  initializeCapstoneOptWInstrsPass(*PR);
  initializeCapstoneFoldMemOffsetPass(*PR);
  initializeCapstonePreRAExpandPseudoPass(*PR);
  initializeCapstoneExpandPseudoPass(*PR);
  initializeCapstoneVectorPeepholePass(*PR);
  initializeCapstoneVLOptimizerPass(*PR);
  initializeCapstoneVMV0EliminationPass(*PR);
  initializeCapstoneInsertVSETVLIPass(*PR);
  initializeCapstoneInsertReadWriteCSRPass(*PR);
  initializeCapstoneInsertWriteVXRMPass(*PR);
  initializeCapstoneDAGToDAGISelLegacyPass(*PR);
  initializeCapstoneMoveMergePass(*PR);
  initializeCapstonePushPopOptPass(*PR);
  initializeCapstoneIndirectBranchTrackingPass(*PR);
  initializeCapstoneLoadStoreOptPass(*PR);
  initializeCapstoneExpandAtomicPseudoPass(*PR);
  initializeCapstoneRedundantCopyEliminationPass(*PR);
  initializeCapstoneAsmPrinterPass(*PR);
}

static Reloc::Model getEffectiveRelocModel(const Triple &TT,
                                           std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

CapstoneTargetMachine::CapstoneTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(
          T, TT.computeDataLayout(Options.MCOptions.getABIName()), TT, CPU, FS,
          Options, getEffectiveRelocModel(TT, RM),
          getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<CapstoneELFTargetObjectFile>()) {
  initAsmInfo();

  // Capstone supports the MachineOutliner.
  setMachineOutliner(true);
  setSupportsDefaultOutlining(true);

  // Capstone supports the debug entry values.
  setSupportsDebugEntryValues(true);

  if (TT.isOSFuchsia() && !TT.isArch64Bit())
    report_fatal_error("Fuchsia is only supported for 64-bit");

  setCFIFixup(true);
}

const CapstoneSubtarget *
CapstoneTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute TuneAttr = F.getFnAttribute("tune-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  std::string TuneCPU =
      TuneAttr.isValid() ? TuneAttr.getValueAsString().str() : CPU;
  std::string FS =
      FSAttr.isValid() ? FSAttr.getValueAsString().str() : TargetFS;

  unsigned RVVBitsMin = RVVVectorBitsMinOpt;
  unsigned RVVBitsMax = RVVVectorBitsMaxOpt;

  Attribute VScaleRangeAttr = F.getFnAttribute(Attribute::VScaleRange);
  if (VScaleRangeAttr.isValid()) {
    if (!RVVVectorBitsMinOpt.getNumOccurrences())
      RVVBitsMin = VScaleRangeAttr.getVScaleRangeMin() * Capstone::RVVBitsPerBlock;
    std::optional<unsigned> VScaleMax = VScaleRangeAttr.getVScaleRangeMax();
    if (VScaleMax.has_value() && !RVVVectorBitsMaxOpt.getNumOccurrences())
      RVVBitsMax = *VScaleMax * Capstone::RVVBitsPerBlock;
  }

  if (RVVBitsMin != -1U) {
    // FIXME: Change to >= 32 when VLEN = 32 is supported.
    assert((RVVBitsMin == 0 || (RVVBitsMin >= 64 && RVVBitsMin <= 65536 &&
                                isPowerOf2_32(RVVBitsMin))) &&
           "V or Zve* extension requires vector length to be in the range of "
           "64 to 65536 and a power 2!");
    assert((RVVBitsMax >= RVVBitsMin || RVVBitsMax == 0) &&
           "Minimum V extension vector length should not be larger than its "
           "maximum!");
  }
  assert((RVVBitsMax == 0 || (RVVBitsMax >= 64 && RVVBitsMax <= 65536 &&
                              isPowerOf2_32(RVVBitsMax))) &&
         "V or Zve* extension requires vector length to be in the range of "
         "64 to 65536 and a power 2!");

  if (RVVBitsMin != -1U) {
    if (RVVBitsMax != 0) {
      RVVBitsMin = std::min(RVVBitsMin, RVVBitsMax);
      RVVBitsMax = std::max(RVVBitsMin, RVVBitsMax);
    }

    RVVBitsMin = llvm::bit_floor(
        (RVVBitsMin < 64 || RVVBitsMin > 65536) ? 0 : RVVBitsMin);
  }
  RVVBitsMax =
      llvm::bit_floor((RVVBitsMax < 64 || RVVBitsMax > 65536) ? 0 : RVVBitsMax);

  SmallString<512> Key;
  raw_svector_ostream(Key) << "RVVMin" << RVVBitsMin << "RVVMax" << RVVBitsMax
                           << CPU << TuneCPU << FS;
  auto &I = SubtargetMap[Key];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    auto ABIName = Options.MCOptions.getABIName();
    if (const MDString *ModuleTargetABI = dyn_cast_or_null<MDString>(
            F.getParent()->getModuleFlag("target-abi"))) {
      auto TargetABI = CapstoneABI::getTargetABI(ABIName);
      if (TargetABI != CapstoneABI::ABI_Unknown &&
          ModuleTargetABI->getString() != ABIName) {
        report_fatal_error("-target-abi option != target-abi module flag");
      }
      ABIName = ModuleTargetABI->getString();
    }
    I = std::make_unique<CapstoneSubtarget>(
        TargetTriple, CPU, TuneCPU, FS, ABIName, RVVBitsMin, RVVBitsMax, *this);
  }
  return I.get();
}

MachineFunctionInfo *CapstoneTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return CapstoneMachineFunctionInfo::create<CapstoneMachineFunctionInfo>(
      Allocator, F, static_cast<const CapstoneSubtarget *>(STI));
}

TargetTransformInfo
CapstoneTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<CapstoneTTIImpl>(this, F));
}

// A Capstone hart has a single byte-addressable address space of 2^XLEN bytes
// for all memory accesses, so it is reasonable to assume that an
// implementation has no-op address space casts. If an implementation makes a
// change to this, they can override it here.
bool CapstoneTargetMachine::isNoopAddrSpaceCast(unsigned SrcAS,
                                             unsigned DstAS) const {
  return true;
}

ScheduleDAGInstrs *
CapstoneTargetMachine::createMachineScheduler(MachineSchedContext *C) const {
  const CapstoneSubtarget &ST = C->MF->getSubtarget<CapstoneSubtarget>();
  ScheduleDAGMILive *DAG = createSchedLive(C);

  if (ST.enableMISchedLoadClustering())
    DAG->addMutation(createLoadClusterDAGMutation(
        DAG->TII, DAG->TRI, /*ReorderWhileClustering=*/true));

  if (ST.enableMISchedStoreClustering())
    DAG->addMutation(createStoreClusterDAGMutation(
        DAG->TII, DAG->TRI, /*ReorderWhileClustering=*/true));

  if (!DisableVectorMaskMutation && ST.hasVInstructions())
    DAG->addMutation(createCapstoneVectorMaskDAGMutation(DAG->TRI));

  return DAG;
}

ScheduleDAGInstrs *
CapstoneTargetMachine::createPostMachineScheduler(MachineSchedContext *C) const {
  const CapstoneSubtarget &ST = C->MF->getSubtarget<CapstoneSubtarget>();
  ScheduleDAGMI *DAG = createSchedPostRA(C);

  if (ST.enablePostMISchedLoadClustering())
    DAG->addMutation(createLoadClusterDAGMutation(
        DAG->TII, DAG->TRI, /*ReorderWhileClustering=*/true));

  if (ST.enablePostMISchedStoreClustering())
    DAG->addMutation(createStoreClusterDAGMutation(
        DAG->TII, DAG->TRI, /*ReorderWhileClustering=*/true));

  return DAG;
}

namespace {

class RVVRegisterRegAlloc : public RegisterRegAllocBase<RVVRegisterRegAlloc> {
public:
  RVVRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : RegisterRegAllocBase(N, D, C) {}
};

static bool onlyAllocateRVVReg(const TargetRegisterInfo &TRI,
                               const MachineRegisterInfo &MRI,
                               const Register Reg) {
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  return CapstoneRegisterInfo::isRVVRegClass(RC);
}

static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }

static llvm::once_flag InitializeDefaultRVVRegisterAllocatorFlag;

/// -capstone-rvv-regalloc=<fast|basic|greedy> command line option.
/// This option could designate the rvv register allocator only.
/// For example: -capstone-rvv-regalloc=basic
static cl::opt<RVVRegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<RVVRegisterRegAlloc>>
    RVVRegAlloc("capstone-rvv-regalloc", cl::Hidden,
                cl::init(&useDefaultRegisterAllocator),
                cl::desc("Register allocator to use for RVV register."));

static void initializeDefaultRVVRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = RVVRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = RVVRegAlloc;
    RVVRegisterRegAlloc::setDefault(RVVRegAlloc);
  }
}

static FunctionPass *createBasicRVVRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateRVVReg);
}

static FunctionPass *createGreedyRVVRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateRVVReg);
}

static FunctionPass *createFastRVVRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateRVVReg, false);
}

static RVVRegisterRegAlloc basicRegAllocRVVReg("basic",
                                               "basic register allocator",
                                               createBasicRVVRegisterAllocator);
static RVVRegisterRegAlloc
    greedyRegAllocRVVReg("greedy", "greedy register allocator",
                         createGreedyRVVRegisterAllocator);

static RVVRegisterRegAlloc fastRegAllocRVVReg("fast", "fast register allocator",
                                              createFastRVVRegisterAllocator);

class CapstonePassConfig : public TargetPassConfig {
public:
  CapstonePassConfig(CapstoneTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    if (TM.getOptLevel() != CodeGenOptLevel::None)
      substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
    setEnableSinkAndFold(EnableSinkFold);
    EnableLoopTermFold = true;
  }

  CapstoneTargetMachine &getCapstoneTargetMachine() const {
    return getTM<CapstoneTargetMachine>();
  }

  void addIRPasses() override;
  bool addPreISel() override;
  void addCodeGenPrepare() override;
  bool addInstSelector() override;
  bool addIRTranslator() override;
  void addPreLegalizeMachineIR() override;
  bool addLegalizeMachineIR() override;
  void addPreRegBankSelect() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
  void addPreEmitPass() override;
  void addPreEmitPass2() override;
  void addPreSched2() override;
  void addMachineSSAOptimization() override;
  FunctionPass *createRVVRegAllocPass(bool Optimized);
  bool addRegAssignAndRewriteFast() override;
  bool addRegAssignAndRewriteOptimized() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addFastRegAlloc() override;
  bool addILPOpts() override;

  std::unique_ptr<CSEConfigBase> getCSEConfig() const override;
};
} // namespace

TargetPassConfig *CapstoneTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new CapstonePassConfig(*this, PM);
}

std::unique_ptr<CSEConfigBase> CapstonePassConfig::getCSEConfig() const {
  return getStandardCSEConfigForOpt(TM->getOptLevel());
}

FunctionPass *CapstonePassConfig::createRVVRegAllocPass(bool Optimized) {
  // Initialize the global default.
  llvm::call_once(InitializeDefaultRVVRegisterAllocatorFlag,
                  initializeDefaultRVVRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = RVVRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  if (Optimized)
    return createGreedyRVVRegisterAllocator();

  return createFastRVVRegisterAllocator();
}

bool CapstonePassConfig::addRegAssignAndRewriteFast() {
  addPass(createRVVRegAllocPass(false));
  addPass(createCapstoneInsertVSETVLIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      EnableCapstoneDeadRegisterElimination)
    addPass(createCapstoneDeadRegisterDefinitionsPass());
  return TargetPassConfig::addRegAssignAndRewriteFast();
}

bool CapstonePassConfig::addRegAssignAndRewriteOptimized() {
  addPass(createRVVRegAllocPass(true));
  addPass(createVirtRegRewriter(false));
  addPass(createCapstoneInsertVSETVLIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      EnableCapstoneDeadRegisterElimination)
    addPass(createCapstoneDeadRegisterDefinitionsPass());
  return TargetPassConfig::addRegAssignAndRewriteOptimized();
}

void CapstonePassConfig::addIRPasses() {
  addPass(createAtomicExpandLegacyPass());
  addPass(createCapstoneZacasABIFixPass());

  if (getOptLevel() != CodeGenOptLevel::None) {
    if (EnableLoopDataPrefetch)
      addPass(createLoopDataPrefetchPass());

    addPass(createCapstoneGatherScatterLoweringPass());
    addPass(createInterleavedAccessPass());
    addPass(createCapstoneCodeGenPreparePass());
  }

  TargetPassConfig::addIRPasses();
}

bool CapstonePassConfig::addPreISel() {
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    // Add a barrier before instruction selection so that we will not get
    // deleted block address after enabling default outlining. See D99707 for
    // more details.
    addPass(createBarrierNoopPass());
  }

  if ((TM->getOptLevel() != CodeGenOptLevel::None &&
       EnableGlobalMerge == cl::BOU_UNSET) ||
      EnableGlobalMerge == cl::BOU_TRUE) {
    // FIXME: Like AArch64, we disable extern global merging by default due to
    // concerns it might regress some workloads. Unlike AArch64, we don't
    // currently support enabling the pass in an "OnlyOptimizeForSize" mode.
    // Investigating and addressing both items are TODO.
    addPass(createGlobalMergePass(TM, /* MaxOffset */ 2047,
                                  /* OnlyOptimizeForSize */ false,
                                  /* MergeExternalByDefault */ true));
  }

  return false;
}

void CapstonePassConfig::addCodeGenPrepare() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createTypePromotionLegacyPass());
  TargetPassConfig::addCodeGenPrepare();
}

bool CapstonePassConfig::addInstSelector() {
  addPass(createCapstoneISelDag(getCapstoneTargetMachine(), getOptLevel()));

  return false;
}

bool CapstonePassConfig::addIRTranslator() {
  addPass(new IRTranslator(getOptLevel()));
  return false;
}

void CapstonePassConfig::addPreLegalizeMachineIR() {
  if (getOptLevel() == CodeGenOptLevel::None) {
    addPass(createCapstoneO0PreLegalizerCombiner());
  } else {
    addPass(createCapstonePreLegalizerCombiner());
  }
}

bool CapstonePassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

void CapstonePassConfig::addPreRegBankSelect() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createCapstonePostLegalizerCombiner());
}

bool CapstonePassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

bool CapstonePassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect(getOptLevel()));
  return false;
}

void CapstonePassConfig::addPreSched2() {
  addPass(createCapstonePostRAExpandPseudoPass());

  // Emit KCFI checks for indirect calls.
  addPass(createKCFIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None)
    addPass(createCapstoneLoadStoreOptPass());
}

void CapstonePassConfig::addPreEmitPass() {
  // TODO: It would potentially be better to schedule copy propagation after
  // expanding pseudos (in addPreEmitPass2). However, performing copy
  // propagation after the machine outliner (which runs after addPreEmitPass)
  // currently leads to incorrect code-gen, where copies to registers within
  // outlined functions are removed erroneously.
  if (TM->getOptLevel() >= CodeGenOptLevel::Default &&
      EnableCapstoneCopyPropagation)
    addPass(createMachineCopyPropagationPass(true));
  if (TM->getOptLevel() >= CodeGenOptLevel::Default)
    addPass(createCapstoneLateBranchOptPass());
  // The IndirectBranchTrackingPass inserts lpad and could have changed the
  // basic block alignment. It must be done before Branch Relaxation to
  // prevent the adjusted offset exceeding the branch range.
  addPass(createCapstoneIndirectBranchTrackingPass());
  addPass(&BranchRelaxationPassID);
  addPass(createCapstoneMakeCompressibleOptPass());
}

void CapstonePassConfig::addPreEmitPass2() {
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    addPass(createCapstoneMoveMergePass());
    // Schedule PushPop Optimization before expansion of Pseudo instruction,
    // ensuring return instruction is detected correctly.
    addPass(createCapstonePushPopOptimizationPass());
  }
  addPass(createCapstoneExpandPseudoPass());

  // Schedule the expansion of AMOs at the last possible moment, avoiding the
  // possibility for other passes to break the requirements for forward
  // progress in the LR/SC block.
  addPass(createCapstoneExpandAtomicPseudoPass());

  // KCFI indirect call checks are lowered to a bundle.
  addPass(createUnpackMachineBundles([&](const MachineFunction &MF) {
    return MF.getFunction().getParent()->getModuleFlag("kcfi");
  }));
}

void CapstonePassConfig::addMachineSSAOptimization() {
  addPass(createCapstoneVectorPeepholePass());
  addPass(createCapstoneFoldMemOffsetPass());

  TargetPassConfig::addMachineSSAOptimization();

  if (TM->getTargetTriple().isCapstone64()) {
    addPass(createCapstoneOptWInstrsPass());
  }
}

void CapstonePassConfig::addPreRegAlloc() {
  addPass(createCapstonePreRAExpandPseudoPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    addPass(createCapstoneMergeBaseOffsetOptPass());
    addPass(createCapstoneVLOptimizerPass());
  }

  addPass(createCapstoneInsertReadWriteCSRPass());
  addPass(createCapstoneInsertWriteVXRMPass());
  addPass(createCapstoneLandingPadSetupPass());

  if (TM->getOptLevel() != CodeGenOptLevel::None && EnableMachinePipeliner)
    addPass(&MachinePipelinerID);

  addPass(createCapstoneVMV0EliminationPass());
}

void CapstonePassConfig::addFastRegAlloc() {
  addPass(&InitUndefID);
  TargetPassConfig::addFastRegAlloc();
}


void CapstonePassConfig::addPostRegAlloc() {
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      EnableRedundantCopyElimination)
    addPass(createCapstoneRedundantCopyEliminationPass());
}

bool CapstonePassConfig::addILPOpts() {
  if (EnableMachineCombiner)
    addPass(&MachineCombinerID);

  return true;
}

void CapstoneTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  PB.registerLateLoopOptimizationsEPCallback([=](LoopPassManager &LPM,
                                                 OptimizationLevel Level) {
    if (Level != OptimizationLevel::O0)
      LPM.addPass(LoopIdiomVectorizePass(LoopIdiomVectorizeStyle::Predicated));
  });
}

yaml::MachineFunctionInfo *
CapstoneTargetMachine::createDefaultFuncInfoYAML() const {
  return new yaml::CapstoneMachineFunctionInfo();
}

yaml::MachineFunctionInfo *
CapstoneTargetMachine::convertFuncInfoToYAML(const MachineFunction &MF) const {
  const auto *MFI = MF.getInfo<CapstoneMachineFunctionInfo>();
  return new yaml::CapstoneMachineFunctionInfo(*MFI);
}

bool CapstoneTargetMachine::parseMachineFunctionInfo(
    const yaml::MachineFunctionInfo &MFI, PerFunctionMIParsingState &PFS,
    SMDiagnostic &Error, SMRange &SourceRange) const {
  const auto &YamlMFI =
      static_cast<const yaml::CapstoneMachineFunctionInfo &>(MFI);
  PFS.MF.getInfo<CapstoneMachineFunctionInfo>()->initializeBaseYamlFields(YamlMFI);
  return false;
}
