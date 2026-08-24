//===--- Capstone.h - Declare Capstone target feature support --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares Capstone TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_CAPSTONE_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_CAPSTONE_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/CapstoneISAInfo.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

namespace clang {
namespace targets {

// Capstone Target
class CapstoneTargetInfo : public TargetInfo {
protected:
  std::string ABI, CPU;
  std::unique_ptr<llvm::CapstoneISAInfo> ISAInfo;

private:
  bool FastScalarUnalignedAccess;
  bool HasExperimental = false;

public:
  CapstoneTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    BFloat16Width = 16;
    BFloat16Align = 16;
    BFloat16Format = &llvm::APFloat::BFloat();
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    LongDoubleFormat = &llvm::APFloat::IEEEquad();
    SuitableAlign = 128;
    WCharType = SignedInt;
    WIntType = UnsignedInt;
    HasRISCVVTypes = true;
    MCountName = "_mcount";
    HasFloat16 = true;
    HasStrictFP = true;
  }

  bool setCPU(const std::string &Name) override {
    if (!isValidCPUName(Name))
      return false;
    CPU = Name;
    return true;
  }

  StringRef getABI() const override { return ABI; }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  llvm::SmallVector<Builtin::InfosShard> getTargetBuiltins() const override;

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  std::string_view getClobbers() const override { return ""; }

  StringRef getConstraintRegister(StringRef Constraint,
                                  StringRef Expression) const override {
    return Expression;
  }

  ArrayRef<const char *> getGCCRegNames() const override;

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0)
      return 10;
    else if (RegNo == 1)
      return 11;
    else
      return -1;
  }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override;

  std::string convertConstraint(const char *&Constraint) const override;

  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override;

  std::optional<std::pair<unsigned, unsigned>>
  getVScaleRange(const LangOptions &LangOpts, ArmStreamingKind Mode,
                 llvm::StringMap<bool> *FeatureMap = nullptr) const override;

  bool hasFeature(StringRef Feature) const override;

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;

  bool hasBitIntType() const override { return true; }

  bool hasBFloat16Type() const override { return true; }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override;

  bool useFP16ConversionIntrinsics() const override {
    return false;
  }

  bool isValidCPUName(StringRef Name) const override;
  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;
  bool isValidTuneCPUName(StringRef Name) const override;
  void fillValidTuneCPUList(SmallVectorImpl<StringRef> &Values) const override;
  bool supportsTargetAttributeTune() const override { return true; }
  ParsedTargetAttr parseTargetAttr(StringRef Str) const override;
  llvm::APInt getFMVPriority(ArrayRef<StringRef> Features) const override;

  std::pair<unsigned, unsigned> hardwareInterferenceSizes() const override {
    return std::make_pair(32, 32);
  }

  bool supportsCpuSupports() const override { return getTriple().isOSLinux(); }
  bool supportsCpuIs() const override { return getTriple().isOSLinux(); }
  bool supportsCpuInit() const override { return getTriple().isOSLinux(); }
  bool validateCpuSupports(StringRef Feature) const override;
  bool validateCpuIs(StringRef CPUName) const override;
  bool isValidFeatureName(StringRef Name) const override;

  bool validateGlobalRegisterVariable(StringRef RegName, unsigned RegSize,
                                      bool &HasSizeMismatch) const override;

  bool checkCFProtectionBranchSupported(DiagnosticsEngine &) const override {
    // Always generate Zicfilp lpad insns
    // Non-zicfilp CPUs would read them as NOP
    return true;
  }

  bool
  checkCFProtectionReturnSupported(DiagnosticsEngine &Diags) const override {
    if (ISAInfo->hasExtension("zimop"))
      return true;
    return TargetInfo::checkCFProtectionReturnSupported(Diags);
  }

  CFBranchLabelSchemeKind getDefaultCFBranchLabelScheme() const override {
    return CFBranchLabelSchemeKind::FuncSig;
  }

  bool
  checkCFBranchLabelSchemeSupported(const CFBranchLabelSchemeKind Scheme,
                                    DiagnosticsEngine &Diags) const override {
    switch (Scheme) {
    case CFBranchLabelSchemeKind::Default:
    case CFBranchLabelSchemeKind::Unlabeled:
    case CFBranchLabelSchemeKind::FuncSig:
      return true;
    }
    return TargetInfo::checkCFBranchLabelSchemeSupported(Scheme, Diags);
  }
};
class LLVM_LIBRARY_VISIBILITY Capstone32TargetInfo : public CapstoneTargetInfo {
public:
  Capstone32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : CapstoneTargetInfo(Triple, Opts) {
    IntPtrType = SignedInt;
    PtrDiffType = SignedInt;
    SizeType = UnsignedInt;
    resetDataLayout("e-m:e-p:32:32-i64:64-n32-S128");
  }

  bool setABI(const std::string &Name) override {
    if (Name == "ilp32e") {
      ABI = Name;
      resetDataLayout("e-m:e-p:32:32-i64:64-n32-S32");
      return true;
    }

    if (Name == "ilp32" || Name == "ilp32f" || Name == "ilp32d") {
      ABI = Name;
      return true;
    }
    return false;
  }

  void setMaxAtomicWidth() override {
    MaxAtomicPromoteWidth = 128;

    if (ISAInfo->hasExtension("a"))
      MaxAtomicInlineWidth = 32;
  }
};

// Map the "Default" address space (0 in C) to LLVM IR address space 200.
// This makes all pointers "Fat Pointers" by default.
const LangASMap CapstoneAddrSpaceMap = {
  200, // Default
  0,   // opencl_global
  0,   // opencl_local
  0,   // opencl_constant
  0,   // opencl_private
  0,   // opencl_generic
  0,   // opencl_global_device
  0,   // opencl_global_host
  0,   // cuda_device
  0,   // cuda_constant
  0,   // cuda_shared
  0,   // sycl_global
  0,   // sycl_global_device
  0,   // sycl_global_host
  0,   // sycl_local
  0,   // sycl_private
  0,   // ptr32_sptr
  0,   // ptr32_uptr
  0,   // ptr64
  0,   // hlsl_groupshared
  0,   // wasm_funcref (unused)
};

class LLVM_LIBRARY_VISIBILITY Capstone64TargetInfo : public CapstoneTargetInfo {
public:
  Capstone64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : CapstoneTargetInfo(Triple, Opts) {
    LongWidth = LongAlign = 64;
    PointerWidth = PointerAlign = 128;

    IntMaxType = Int64Type = SignedLong;
    AddrSpaceMap = &CapstoneAddrSpaceMap;

    // IMPORTANT: New DataLayout for PureCap mode.
    // p:64:128         -- AS0 is 64-bit but 128-bit aligned
    //                         (Workaround for Clang consistency check)
    // p200:128:128:128:64 -- AS200 is 128-bit (Capabilities), but its ADDRESS
    //                        is 64-bit, so pointer arithmetic stays in i64
    // ni:200           -- Non-Integral pointers! Prevents unsafe optimizations
    // A200/P200/G200   -- Use AS200 as the default address space for
    //                     alloca/stack (A), program (P), and globals (G).
    resetDataLayout(
        "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128"
        "-ni:200-A200-P200-G200");
  }

  // A capability is 128 bits; the address inside it is 64. intptr_t, size_t and
  // ptrdiff_t describe the ADDRESS -- which is what this target's C types
  // already say, and this is how codegen is told the same thing.
  uint64_t getMaxAddressWidth() const override { return 64; }

  bool setABI(const std::string &Name) override {
    if (Name == "lp64e") {
      ABI = Name;
      resetDataLayout("e-m:e-p:64:64-i64:64-i128:128-n32:64-S64");
      return true;
    }

    if (Name == "lp64" || Name == "lp64f" || Name == "lp64d") {
      ABI = Name;
      return true;
    }
    return false;
  }

  // __int128 and wide _BitInt are ORDINARY INTEGERS here, and both were refused until
  // 2026-08-24 because they were not. MVT::i128 used to be the machine type for a
  // CAPABILITY, with no separate capability MVT to hold the two apart, so the backend
  // told them apart by heuristics that defaulted to capability. A source-level
  // __int128 matched none of them and was compiled AS a capability, silently:
  //
  //     unsigned __int128 a + b  ->  cincoffset a0, a1, a0   (a cursor increment)
  //     (unsigned long)(v >> 64) ->  fatal error, "cannot lower a 128-bit right shift"
  //
  // A capability is MVT::c128 in its own register class now, exactly as the note here
  // said lifting this would require, so i128 is an ordinary illegal type that the
  // generic legalizer expands. The output is instruction-for-instruction identical to
  // upstream riscv64 on the same IR. Both overrides are therefore gone and the base
  // rules apply.

  void setMaxAtomicWidth() override {
    MaxAtomicPromoteWidth = 128;

    if (ISAInfo->hasExtension("a"))
      MaxAtomicInlineWidth = 64;
  }
};
} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_CAPSTONE_H 