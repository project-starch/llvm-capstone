//===-- CapstoneTargetStreamer.h - Capstone Target Streamer ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_MCTARGETDESC_CapstoneTARGETSTREAMER_H
#define LLVM_LIB_TARGET_Capstone_MCTARGETDESC_CapstoneTARGETSTREAMER_H

#include "Capstone.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {

class formatted_raw_ostream;

enum class CapstoneOptionArchArgType {
  Full,
  Plus,
  Minus,
};

struct CapstoneOptionArchArg {
  CapstoneOptionArchArgType Type;
  std::string Value;

  CapstoneOptionArchArg(CapstoneOptionArchArgType Type, std::string Value)
      : Type(Type), Value(Value) {}
};

class CapstoneTargetStreamer : public MCTargetStreamer {
  CapstoneABI::ABI TargetABI = CapstoneABI::ABI_Unknown;
  bool HasRVC = false;
  bool HasTSO = false;

public:
  CapstoneTargetStreamer(MCStreamer &S);
  void finish() override;
  virtual void reset();

  virtual void emitDirectiveOptionArch(ArrayRef<CapstoneOptionArchArg> Args);
  virtual void emitDirectiveOptionExact();
  virtual void emitDirectiveOptionNoExact();
  virtual void emitDirectiveOptionPIC();
  virtual void emitDirectiveOptionNoPIC();
  virtual void emitDirectiveOptionPop();
  virtual void emitDirectiveOptionPush();
  virtual void emitDirectiveOptionRelax();
  virtual void emitDirectiveOptionNoRelax();
  virtual void emitDirectiveOptionRVC();
  virtual void emitDirectiveOptionNoRVC();
  virtual void emitDirectiveVariantCC(MCSymbol &Symbol);
  virtual void emitAttribute(unsigned Attribute, unsigned Value);
  virtual void finishAttributeSection();
  virtual void emitTextAttribute(unsigned Attribute, StringRef String);
  virtual void emitIntTextAttribute(unsigned Attribute, unsigned IntValue,
                                    StringRef StringValue);
  void emitNoteGnuPropertySection(const uint32_t Feature1And);

  void emitTargetAttributes(const MCSubtargetInfo &STI, bool EmitStackAlign);
  void setTargetABI(CapstoneABI::ABI ABI);
  CapstoneABI::ABI getTargetABI() const { return TargetABI; }
  void setFlagsFromFeatures(const MCSubtargetInfo &STI);
  bool hasRVC() const { return HasRVC; }
  bool hasTSO() const { return HasTSO; }
};

// This part is for ascii assembly output
class CapstoneTargetAsmStreamer : public CapstoneTargetStreamer {
  formatted_raw_ostream &OS;

  void finishAttributeSection() override;
  void emitAttribute(unsigned Attribute, unsigned Value) override;
  void emitTextAttribute(unsigned Attribute, StringRef String) override;
  void emitIntTextAttribute(unsigned Attribute, unsigned IntValue,
                            StringRef StringValue) override;

public:
  CapstoneTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);

  void emitDirectiveOptionArch(ArrayRef<CapstoneOptionArchArg> Args) override;
  void emitDirectiveOptionExact() override;
  void emitDirectiveOptionNoExact() override;
  void emitDirectiveOptionPIC() override;
  void emitDirectiveOptionNoPIC() override;
  void emitDirectiveOptionPop() override;
  void emitDirectiveOptionPush() override;
  void emitDirectiveOptionRelax() override;
  void emitDirectiveOptionNoRelax() override;
  void emitDirectiveOptionRVC() override;
  void emitDirectiveOptionNoRVC() override;
  void emitDirectiveVariantCC(MCSymbol &Symbol) override;
};

}
#endif
