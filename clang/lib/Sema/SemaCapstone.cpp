//===------ SemaCapstone.cpp ---- Capstone target-specific routines -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis functions specific to Capstone.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaCapstone.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Sema.h"

using namespace clang;

SemaCapstone::SemaCapstone(Sema &S) : SemaBase(S) {}

void SemaCapstone::checkPointerRoundTrip(Expr *Src, QualType DestTy,
                                         SourceRange OpRange) {
  if (!DestTy->isPointerType() || !Src->getType()->isIntegerType())
    return;
  // (a) `(T *)(integer)p`: the integer is an explicit cast of a pointer.
  if (const auto *CE = dyn_cast<CastExpr>(Src->IgnoreParenImpCasts())) {
    if (CE->getSubExpr()->IgnoreParenImpCasts()->getType()->isPointerType()) {
      Diag(OpRange.getBegin(), diag::warn_capstone_pointer_roundtrip)
          << 0 << Src->getType() << DestTy << Src->getSourceRange();
      return;
    }
  }
  // (b) `(T *)x` with x of a type spelled uintptr_t / intptr_t: the typedef's
  // whole purpose is to hold a pointer, and on this target it cannot.
  for (QualType T = Src->getType();;) {
    const auto *TT = T->getAs<TypedefType>();
    if (!TT)
      break;
    StringRef Name = TT->getDecl()->getName();
    if (Name == "uintptr_t" || Name == "intptr_t") {
      Diag(OpRange.getBegin(), diag::warn_capstone_pointer_roundtrip)
          << 1 << Src->getType() << DestTy << Src->getSourceRange();
      return;
    }
    T = TT->desugar();
  }
}

// The capability CSR ids ccsrrw can name. QEMU's helper_csccsrrw switches on
// exactly these (capstone-qemu target/riscv/capstone_defs.h:32-49: ctvec 0,
// cih 1, cepc 2, cscratch 4, and the cpmp entries, id & 0xfff0 == 0x10) and
// ASSERTS on anything else -- an emulator abort, not a guest fault -- so the
// front end is the only place a wrong id is caught cleanly. 3 is reserved.
static bool isCapstoneCCSRId(uint64_t Id) {
  return Id == 0 || Id == 1 || Id == 2 || Id == 4 || (Id & 0xfff0) == 0x10;
}

bool SemaCapstone::CheckBuiltinFunctionCall(const TargetInfo &TI,
                                            unsigned BuiltinID,
                                            CallExpr *TheCall) {
  switch (BuiltinID) {
  default:
    return false;

  // TIGHTEN's immediate is a permission mask. The encoding field is five bits,
  // but a permission is three (the R/W/X bits); the RTL raises
  // ILLEGAL_OPERAND_VALUE for imm > 7 (capstone_dyn_unit.anvil:231-232), while
  // the spec and QEMU clamp it to no permissions. 0..7 is right on all three.
  case Capstone::BI__builtin_capstone_cap_tighten:
    return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 7);

  case Capstone::BI__builtin_capstone_cap_ccsrrw: {
    llvm::APSInt Result;
    if (SemaRef.BuiltinConstantArg(TheCall, 1, Result))
      return true;
    if (Result.isNegative() || !isCapstoneCCSRId(Result.getZExtValue()))
      return Diag(TheCall->getArg(1)->getBeginLoc(),
                  diag::err_capstone_builtin_invalid_ccsr)
             << toString(Result, 10) << TheCall->getArg(1)->getSourceRange();
    return false;
  }

  // SHRINK with constant bounds: base must be below end, or the instruction
  // raises ILLEGAL_OPERAND_VALUE on every implementation (a zero-size object
  // is illegal too -- C-34).
  case Capstone::BI__builtin_capstone_cap_shrink: {
    Expr *BaseE = TheCall->getArg(1), *EndE = TheCall->getArg(2);
    Expr::EvalResult Base, End;
    if (!BaseE->isValueDependent() && !EndE->isValueDependent() &&
        BaseE->EvaluateAsInt(Base, getASTContext()) &&
        EndE->EvaluateAsInt(End, getASTContext()) &&
        Base.Val.getInt().getZExtValue() >= End.Val.getInt().getZExtValue())
      return Diag(BaseE->getBeginLoc(), diag::err_capstone_builtin_shrink_bounds)
             << toString(Base.Val.getInt(), 10) << toString(End.Val.getInt(), 10)
             << BaseE->getSourceRange() << EndE->getSourceRange();
    return false;
  }

  // Inherited from the RISCV copy: the scalar-crypto immediates, with the
  // ranges SemaRISCV::CheckBuiltinFunctionCall enforces.
  case Capstone::BI__builtin_capstone_aes32dsi:
  case Capstone::BI__builtin_capstone_aes32dsmi:
  case Capstone::BI__builtin_capstone_aes32esi:
  case Capstone::BI__builtin_capstone_aes32esmi:
  case Capstone::BI__builtin_capstone_sm4ks:
  case Capstone::BI__builtin_capstone_sm4ed:
    return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 3);
  case Capstone::BI__builtin_capstone_aes64ks1i:
    return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 10);
  }
}
