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

// How many pointer-to-integer casts an integer expression is built from,
// looking through the operators an address computation uses. One means the
// value is an address computed from one pointer right here, which the backend
// turns back into that pointer moved (CapstoneRecoverProvenance).
//
// The answer has to agree with what that pass will do, or the warning goes
// silent on a round trip that then traps untagged. Two shapes carry a pointer
// and are still counted as unrecoverable, because the pass declines them: an
// address that flows INTO a multiply, shift, divide or remainder is not that
// pointer's address moved any more, and a conditional whose arms are not both
// the same one pointer would give that pointer's authority at a foreign
// address. Which arm holds which pointer is not decidable here, so any pointer
// inside a conditional disqualifies it: `c ? (uintptr_t)p : (uintptr_t)q`
// counts two pointers in the pass and would count one here. A warning one time
// too many beats a trap with no warning at all.
//
// Still silent, and not fixable in the front end: a round trip the pass declines
// because the pointer may be NULL (it may not move a base that holds no
// capability). Whether a pointer can be null is not visible here.
static constexpr unsigned Disqualified = 2; // any count but one

static unsigned countPointerSources(const Expr *E) {
  E = E->IgnoreParens();
  if (const auto *CE = dyn_cast<CastExpr>(E)) {
    if (CE->getSubExpr()->IgnoreParens()->getType()->isPointerType())
      return 1;
    return countPointerSources(CE->getSubExpr());
  }
  if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
    switch (BO->getOpcode()) {
    case BO_Add: case BO_Sub: case BO_And: case BO_Or: case BO_Xor:
      return countPointerSources(BO->getLHS()) +
             countPointerSources(BO->getRHS());
    case BO_Mul: case BO_Shl: case BO_Shr: case BO_Div: case BO_Rem: {
      // On plain integers these are fine and common -- `(uintptr_t)p + i * 8`
      // keeps its one source through the multiply's operands. With an address
      // inside one, the result is not an address the pass will rebuild.
      unsigned N = countPointerSources(BO->getLHS()) +
                   countPointerSources(BO->getRHS());
      return N ? Disqualified : 0;
    }
    default:
      return 0;
    }
  }
  if (const auto *UO = dyn_cast<UnaryOperator>(E))
    return countPointerSources(UO->getSubExpr());
  if (const auto *CO = dyn_cast<ConditionalOperator>(E)) {
    unsigned N = countPointerSources(CO->getTrueExpr()) +
                 countPointerSources(CO->getFalseExpr());
    return N ? Disqualified : 0;
  }
  return 0;
}

void SemaCapstone::checkPointerRoundTrip(Expr *Src, QualType DestTy,
                                         SourceRange OpRange) {
  if (!DestTy->isPointerType() || !Src->getType()->isIntegerType())
    return;
  // An integer computed in this very expression from one pointer --
  // `(T *)(uintptr_t)p`, `(T *)(((uintptr_t)p + 15) & ~15)` -- is not
  // diagnosed: the backend (CapstoneRecoverProvenance) rebuilds the result from
  // that pointer's capability.
  //
  // `(T *)x` with x of a type spelled uintptr_t / intptr_t: the typedef's whole
  // purpose is to hold a pointer, and on this target it can hold only the
  // address. Whether this cast gets the capability back depends on where x came
  // from -- computed from one pointer in the same function, yes; loaded from a
  // struct field or passed in, no -- which the front end cannot see, so the
  // warning says which case is safe.
  if (countPointerSources(Src) == 1)
    return; // e.g. (T *)(((uintptr_t)p + 15) & ~15): computed right here.
  for (QualType T = Src->getType();;) {
    const auto *TT = T->getAs<TypedefType>();
    if (!TT)
      break;
    StringRef Name = TT->getDecl()->getName();
    if (Name == "uintptr_t" || Name == "intptr_t") {
      Diag(OpRange.getBegin(), diag::warn_capstone_pointer_roundtrip)
          << Src->getType() << DestTy << Src->getSourceRange();
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
