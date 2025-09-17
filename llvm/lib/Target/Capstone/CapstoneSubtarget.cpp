//===-- CapstoneSubtarget.cpp - Capstone Subtarget Information -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Capstone specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "CapstoneSubtarget.h"
#include "GISel/CapstoneCallLowering.h"
#include "GISel/CapstoneLegalizerInfo.h"
#include "Capstone.h"
#include "CapstoneFrameLowering.h"
#include "CapstoneSelectionDAGInfo.h"
#include "CapstoneTargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "CapstoneGenSubtargetInfo.inc"

#define GET_Capstone_MACRO_FUSION_PRED_IMPL
#include "CapstoneGenMacroFusion.inc"

namespace llvm::CapstoneTuneInfoTable {

#define GET_CapstoneTuneInfoTable_IMPL
#include "CapstoneGenSearchableTables.inc"
} // namespace llvm::CapstoneTuneInfoTable

static cl::opt<unsigned> RVVVectorLMULMax(
    "capstone-v-fixed-length-vector-lmul-max",
    cl::desc("The maximum LMUL value to use for fixed length vectors. "
             "Fractional LMUL values are not supported."),
    cl::init(8), cl::Hidden);

static cl::opt<bool> CapstoneDisableUsingConstantPoolForLargeInts(
    "capstone-disable-using-constant-pool-for-large-ints",
    cl::desc("Disable using constant pool for large integers."),
    cl::init(false), cl::Hidden);

static cl::opt<unsigned> CapstoneMaxBuildIntsCost(
    "capstone-max-build-ints-cost",
    cl::desc("The maximum cost used for building integers."), cl::init(0),
    cl::Hidden);

static cl::opt<bool> UseAA("capstone-use-aa", cl::init(true),
                           cl::desc("Enable the use of AA during codegen."));

static cl::opt<unsigned> CapstoneMinimumJumpTableEntries(
    "capstone-min-jump-table-entries", cl::Hidden,
    cl::desc("Set minimum number of entries to use a jump table on Capstone"));

static cl::opt<bool> UseMIPSLoadStorePairsOpt(
    "use-capstone-mips-load-store-pairs",
    cl::desc("Enable the load/store pair optimization pass"), cl::init(false),
    cl::Hidden);

static cl::opt<bool> UseCCMovInsn("use-capstone-ccmov",
                                  cl::desc("Use 'mips.ccmov' instruction"),
                                  cl::init(true), cl::Hidden);

void CapstoneSubtarget::anchor() {}

CapstoneSubtarget &
CapstoneSubtarget::initializeSubtargetDependencies(const Triple &TT, StringRef CPU,
                                                StringRef TuneCPU, StringRef FS,
                                                StringRef ABIName) {
  // Determine default and user-specified characteristics
  bool Is64Bit = TT.isArch64Bit();
  if (CPU.empty() || CPU == "generic")
    CPU = Is64Bit ? "generic-rv64" : "generic-rv32";

  if (TuneCPU.empty())
    TuneCPU = CPU;

  TuneInfo = CapstoneTuneInfoTable::getCapstoneTuneInfo(TuneCPU);
  // If there is no TuneInfo for this CPU, we fail back to generic.
  if (!TuneInfo)
    TuneInfo = CapstoneTuneInfoTable::getCapstoneTuneInfo("generic");
  assert(TuneInfo && "TuneInfo shouldn't be nullptr!");

  ParseSubtargetFeatures(CPU, TuneCPU, FS);
  TargetABI = CapstoneABI::computeTargetABI(TT, getFeatureBits(), ABIName);
  CapstoneFeatures::validate(TT, getFeatureBits());
  return *this;
}

CapstoneSubtarget::CapstoneSubtarget(const Triple &TT, StringRef CPU,
                               StringRef TuneCPU, StringRef FS,
                               StringRef ABIName, unsigned RVVVectorBitsMin,
                               unsigned RVVVectorBitsMax,
                               const TargetMachine &TM)
    : CapstoneGenSubtargetInfo(TT, CPU, TuneCPU, FS),
      RVVVectorBitsMin(RVVVectorBitsMin), RVVVectorBitsMax(RVVVectorBitsMax),
      FrameLowering(
          initializeSubtargetDependencies(TT, CPU, TuneCPU, FS, ABIName)),
      InstrInfo(*this), RegInfo(getHwMode()), TLInfo(TM, *this) {
  TSInfo = std::make_unique<CapstoneSelectionDAGInfo>();
}

CapstoneSubtarget::~CapstoneSubtarget() = default;

const SelectionDAGTargetInfo *CapstoneSubtarget::getSelectionDAGInfo() const {
  return TSInfo.get();
}

const CallLowering *CapstoneSubtarget::getCallLowering() const {
  if (!CallLoweringInfo)
    CallLoweringInfo.reset(new CapstoneCallLowering(*getTargetLowering()));
  return CallLoweringInfo.get();
}

InstructionSelector *CapstoneSubtarget::getInstructionSelector() const {
  if (!InstSelector) {
    InstSelector.reset(createCapstoneInstructionSelector(
        *static_cast<const CapstoneTargetMachine *>(&TLInfo.getTargetMachine()),
        *this, *getRegBankInfo()));
  }
  return InstSelector.get();
}

const LegalizerInfo *CapstoneSubtarget::getLegalizerInfo() const {
  if (!Legalizer)
    Legalizer.reset(new CapstoneLegalizerInfo(*this));
  return Legalizer.get();
}

const CapstoneRegisterBankInfo *CapstoneSubtarget::getRegBankInfo() const {
  if (!RegBankInfo)
    RegBankInfo.reset(new CapstoneRegisterBankInfo(getHwMode()));
  return RegBankInfo.get();
}

bool CapstoneSubtarget::useConstantPoolForLargeInts() const {
  return !CapstoneDisableUsingConstantPoolForLargeInts;
}

unsigned CapstoneSubtarget::getMaxBuildIntsCost() const {
  // Loading integer from constant pool needs two instructions (the reason why
  // the minimum cost is 2): an address calculation instruction and a load
  // instruction. Usually, address calculation and instructions used for
  // building integers (addi, slli, etc.) can be done in one cycle, so here we
  // set the default cost to (LoadLatency + 1) if no threshold is provided.
  return CapstoneMaxBuildIntsCost == 0
             ? getSchedModel().LoadLatency + 1
             : std::max<unsigned>(2, CapstoneMaxBuildIntsCost);
}

unsigned CapstoneSubtarget::getMaxRVVVectorSizeInBits() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");

  // ZvlLen specifies the minimum required vlen. The upper bound provided by
  // capstone-v-vector-bits-max should be no less than it.
  if (RVVVectorBitsMax != 0 && RVVVectorBitsMax < ZvlLen)
    report_fatal_error("capstone-v-vector-bits-max specified is lower "
                       "than the Zvl*b limitation");

  return RVVVectorBitsMax;
}

unsigned CapstoneSubtarget::getMinRVVVectorSizeInBits() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");

  if (RVVVectorBitsMin == -1U)
    return ZvlLen;

  // ZvlLen specifies the minimum required vlen. The lower bound provided by
  // capstone-v-vector-bits-min should be no less than it.
  if (RVVVectorBitsMin != 0 && RVVVectorBitsMin < ZvlLen)
    report_fatal_error("capstone-v-vector-bits-min specified is lower "
                       "than the Zvl*b limitation");

  return RVVVectorBitsMin;
}

unsigned CapstoneSubtarget::getMaxLMULForFixedLengthVectors() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");
  assert(RVVVectorLMULMax <= 8 &&
         llvm::has_single_bit<uint32_t>(RVVVectorLMULMax) &&
         "V extension requires a LMUL to be at most 8 and a power of 2!");
  return llvm::bit_floor(std::clamp<unsigned>(RVVVectorLMULMax, 1, 8));
}

bool CapstoneSubtarget::useRVVForFixedLengthVectors() const {
  return hasVInstructions() &&
         getMinRVVVectorSizeInBits() >= Capstone::RVVBitsPerBlock;
}

bool CapstoneSubtarget::enableSubRegLiveness() const { return true; }

bool CapstoneSubtarget::enableMachinePipeliner() const {
  return getSchedModel().hasInstrSchedModel();
}

  /// Enable use of alias analysis during code generation (during MI
  /// scheduling, DAGCombine, etc.).
bool CapstoneSubtarget::useAA() const { return UseAA; }

unsigned CapstoneSubtarget::getMinimumJumpTableEntries() const {
  return CapstoneMinimumJumpTableEntries.getNumOccurrences() > 0
             ? CapstoneMinimumJumpTableEntries
             : TuneInfo->MinimumJumpTableEntries;
}

void CapstoneSubtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                         const SchedRegion &Region) const {
  // Do bidirectional scheduling since it provides a more balanced scheduling
  // leading to better performance. This will increase compile time.
  Policy.OnlyTopDown = false;
  Policy.OnlyBottomUp = false;

  // Disabling the latency heuristic can reduce the number of spills/reloads but
  // will cause some regressions on some cores.
  Policy.DisableLatencyHeuristic = DisableLatencySchedHeuristic;

  // Spilling is generally expensive on all Capstone cores, so always enable
  // register-pressure tracking. This will increase compile time.
  Policy.ShouldTrackPressure = true;
}

void CapstoneSubtarget::overridePostRASchedPolicy(
    MachineSchedPolicy &Policy, const SchedRegion &Region) const {
  MISched::Direction PostRASchedDirection = getPostRASchedDirection();
  if (PostRASchedDirection == MISched::TopDown) {
    Policy.OnlyTopDown = true;
    Policy.OnlyBottomUp = false;
  } else if (PostRASchedDirection == MISched::BottomUp) {
    Policy.OnlyTopDown = false;
    Policy.OnlyBottomUp = true;
  } else if (PostRASchedDirection == MISched::Bidirectional) {
    Policy.OnlyTopDown = false;
    Policy.OnlyBottomUp = false;
  }
}

bool CapstoneSubtarget::useLoadStorePairs() const {
  return UseMIPSLoadStorePairsOpt && HasVendorXMIPSLSP;
}

bool CapstoneSubtarget::useCCMovInsn() const {
  return UseCCMovInsn && HasVendorXMIPSCMov;
}
