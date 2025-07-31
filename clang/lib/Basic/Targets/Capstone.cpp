//===--- Capstone.cpp - Implement Capstone target feature support -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Capstone TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "Targets/Capstone.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"

using namespace clang;
using namespace clang::targets;

void CapstoneTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__Capstone__", "1");
}

#define GET_BUILTIN_STR_TABLE
#include "clang/Basic/BuiltinsCapstone.inc"
#undef GET_BUILTIN_STR_TABLE

static constexpr Builtin::Info BuiltinInfos[] = {
#define GET_BUILTIN_INFOS
#include "clang/Basic/BuiltinsCapstone.inc"
#undef GET_BUILTIN_INFOS
};

SmallVector<Builtin::InfosShard> CapstoneTargetInfo::getTargetBuiltins() const {
  return {{&BuiltinStrings, BuiltinInfos}};
}

// From the book:
//
// static constexpr std::array<Builtin::Info, NumCapstoneBuiltins> BuiltinInfo = {
// #define BUILTIN(ID, TYPE, ATTRS)                                               \
//   {#ID, TYPE, ATTRS, nullptr, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
// #define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
//   {#ID, TYPE, ATTRS, nullptr, HEADER, ALL_LANGUAGES},
// #define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
//   {#ID, TYPE, ATTRS, FEATURE, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
// #include "clang/Basic/BuiltinsCapstone.inc"
// };
//
// ArrayRef<Builtin::Info> CapstoneTargetInfo::getTargetBuiltins() const {
//   return llvm::ArrayRef(BuiltinInfo,
//                         clang::Capstone::LastTSBuiltin - Builtin::FirstTSBuiltin);
// }