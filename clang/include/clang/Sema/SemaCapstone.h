//===----- SemaCapstone.h ---- Capstone target-specific routines ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to Capstone.
///
/// Until 2026-09-05 no __builtin_capstone_* builtin had any Sema at all: the
/// SemaChecking dispatch had a riscv case and no capstone case, so an
/// out-of-range immediate reached instruction selection and the compiler died
/// with a backend fatal error and a stack dump (`cap_tighten(p, 999)`).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMACAPSTONE_H
#define LLVM_CLANG_SEMA_SEMACAPSTONE_H

#include "clang/AST/ASTFwd.h"
#include "clang/Sema/SemaBase.h"

namespace clang {
class TargetInfo;

class SemaCapstone : public SemaBase {
public:
  SemaCapstone(Sema &S);

  bool CheckBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                CallExpr *TheCall);

  /// -Wcapstone-pointer-roundtrip: an integer-to-pointer C-style cast whose
  /// integer's type is spelled uintptr_t/intptr_t.  An integer cannot carry a
  /// capability's tag; the backend rebuilds the pointer only when the integer
  /// was computed from one pointer in the same function, so a value that came
  /// from memory or from a caller is untagged.
  void checkPointerRoundTrip(Expr *Src, QualType DestTy, SourceRange OpRange);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMACAPSTONE_H
