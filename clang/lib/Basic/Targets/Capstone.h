//===--- Capstone.h - Declare Capstone target feature support -----*- C++ -*-===//
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h" // For LLVM_LIBRARY_VISIBILITY.
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

// Capstone class
class LLVM_LIBRARY_VISIBILITY CapstoneTargetInfo : public TargetInfo { // book
// class CapstoneTargetInfo : public TargetInfo {
public:
  CapstoneTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    resetDataLayout(
        // Little-endian.
        "e-"
        // Pointer size is 16-bit and the alignment matches.
        "p:16:16:16-"
        // Supports natively 16-bit and 32-bit integer.
        "n16:32-"
        // i64 are aligned on 64, i32 on 32, i16 on 16 and i1 on 8.
        "i64:64:64-i32:32:32-i16:16:16-i1:8:8-"
        // f32 aligned on 32-bit.
        "f32:32:32-"
        // v32 aligned on 32-bit.
        "v32:32:32");

    PointerWidth = 16;
    PointerAlign = 16;
    // IntWidth = 16;
    // IntAlign = 16;
    // LongWidth = 32;
    // LongAlign = 32;
    // LongLongWidth = 64;
    // LongLongAlign = 64;
  }

  /// Appends the target-specific \#define values for this
  /// target set to the specified buffer.
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  /// Return information about target-specific builtins for
  /// the current primary target, and info about which builtins are non-portable
  /// across the current set of primary and secondary targets.
  // ArrayRef<Builtin::Info> getTargetBuiltins() const override; // old, from the book
  SmallVector<Builtin::InfosShard> getTargetBuiltins() const override;

  /// Returns the kind of __builtin_va_list type that should be used
  /// with this target.
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return CharPtrBuiltinVaList;
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return false;
  }

  /// Returns a string of target-specific clobbers, in LLVM format.
  std::string_view getClobbers() const override { return ""; }

  ArrayRef<const char *> getGCCRegNames() const override {
    // return std::nullopt;
    return {};
  }
  ArrayRef<GCCRegAlias> getGCCRegAliases() const override {
    // return std::nullopt;
    return {};
  }
};
} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETS_Capstone_H
