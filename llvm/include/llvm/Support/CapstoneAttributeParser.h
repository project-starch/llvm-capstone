//===-- CapstoneAttributeParser.h - Capstone Attribute Parser ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CapstoneATTRIBUTEPARSER_H
#define LLVM_SUPPORT_CapstoneATTRIBUTEPARSER_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/ELFAttrParserCompact.h"
#include "llvm/Support/CapstoneAttributes.h"

namespace llvm {
class LLVM_ABI CapstoneAttributeParser : public ELFCompactAttrParser {
  struct DisplayHandler {
    CapstoneAttrs::AttrType attribute;
    Error (CapstoneAttributeParser::*routine)(unsigned);
  };
  static const DisplayHandler displayRoutines[];

  Error handler(uint64_t tag, bool &handled) override;

  Error unalignedAccess(unsigned tag);
  Error stackAlign(unsigned tag);
  Error atomicAbi(unsigned tag);

public:
  CapstoneAttributeParser(ScopedPrinter *sw)
      : ELFCompactAttrParser(sw, CapstoneAttrs::getCapstoneAttributeTags(), "capstone") {
  }
  CapstoneAttributeParser()
      : ELFCompactAttrParser(CapstoneAttrs::getCapstoneAttributeTags(), "capstone") {}
};

} // namespace llvm

#endif
