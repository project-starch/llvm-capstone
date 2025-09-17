//===------- CapstoneConstantPoolValue.cpp - Capstone constantpool value -------===//
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

#include "CapstoneConstantPoolValue.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

CapstoneConstantPoolValue::CapstoneConstantPoolValue(Type *Ty, const GlobalValue *GV)
    : MachineConstantPoolValue(Ty), GV(GV), Kind(CapstoneCPKind::GlobalValue) {}

CapstoneConstantPoolValue::CapstoneConstantPoolValue(LLVMContext &C, StringRef S)
    : MachineConstantPoolValue(Type::getInt64Ty(C)), S(S),
      Kind(CapstoneCPKind::ExtSymbol) {}

CapstoneConstantPoolValue *CapstoneConstantPoolValue::Create(const GlobalValue *GV) {
  return new CapstoneConstantPoolValue(GV->getType(), GV);
}

CapstoneConstantPoolValue *CapstoneConstantPoolValue::Create(LLVMContext &C,
                                                       StringRef S) {
  return new CapstoneConstantPoolValue(C, S);
}

int CapstoneConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                      Align Alignment) {
  const std::vector<MachineConstantPoolEntry> &Constants = CP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    if (Constants[i].isMachineConstantPoolEntry() &&
        Constants[i].getAlign() >= Alignment) {
      auto *CPV =
          static_cast<CapstoneConstantPoolValue *>(Constants[i].Val.MachineCPVal);
      if (equals(CPV))
        return i;
    }
  }

  return -1;
}

void CapstoneConstantPoolValue::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  if (isGlobalValue())
    ID.AddPointer(GV);
  else {
    assert(isExtSymbol() && "unrecognized constant pool type");
    ID.AddString(S);
  }
}

void CapstoneConstantPoolValue::print(raw_ostream &O) const {
  if (isGlobalValue())
    O << GV->getName();
  else {
    assert(isExtSymbol() && "unrecognized constant pool type");
    O << S;
  }
}

bool CapstoneConstantPoolValue::equals(const CapstoneConstantPoolValue *A) const {
  if (isGlobalValue() && A->isGlobalValue())
    return GV == A->GV;
  if (isExtSymbol() && A->isExtSymbol())
    return S == A->S;

  return false;
}
