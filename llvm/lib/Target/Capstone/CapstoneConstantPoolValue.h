//===--- CapstoneConstantPoolValue.h - Capstone constantpool value ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Capstone specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_CapstoneCONSTANTPOOLVALUE_H
#define LLVM_LIB_TARGET_Capstone_CapstoneCONSTANTPOOLVALUE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

class BlockAddress;
class GlobalValue;
class LLVMContext;

/// A Capstone-specific constant pool value.
class CapstoneConstantPoolValue : public MachineConstantPoolValue {
  const GlobalValue *GV;
  const StringRef S;

  CapstoneConstantPoolValue(Type *Ty, const GlobalValue *GV);
  CapstoneConstantPoolValue(LLVMContext &C, StringRef S);

private:
  enum class CapstoneCPKind { ExtSymbol, GlobalValue };
  CapstoneCPKind Kind;

public:
  ~CapstoneConstantPoolValue() = default;

  static CapstoneConstantPoolValue *Create(const GlobalValue *GV);
  static CapstoneConstantPoolValue *Create(LLVMContext &C, StringRef S);

  bool isGlobalValue() const { return Kind == CapstoneCPKind::GlobalValue; }
  bool isExtSymbol() const { return Kind == CapstoneCPKind::ExtSymbol; }

  const GlobalValue *getGlobalValue() const { return GV; }
  StringRef getSymbol() const { return S; }

  int getExistingMachineCPValue(MachineConstantPool *CP,
                                Align Alignment) override;

  void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;

  void print(raw_ostream &O) const override;

  bool equals(const CapstoneConstantPoolValue *A) const;
};

} // end namespace llvm

#endif
