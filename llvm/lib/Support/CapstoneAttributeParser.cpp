//===-- CapstoneAttributeParser.cpp - Capstone Attribute Parser -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CapstoneAttributeParser.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

const CapstoneAttributeParser::DisplayHandler
    CapstoneAttributeParser::displayRoutines[] = {
        {
            CapstoneAttrs::ARCH,
            &ELFCompactAttrParser::stringAttribute,
        },
        {
            CapstoneAttrs::PRIV_SPEC,
            &ELFCompactAttrParser::integerAttribute,
        },
        {
            CapstoneAttrs::PRIV_SPEC_MINOR,
            &ELFCompactAttrParser::integerAttribute,
        },
        {
            CapstoneAttrs::PRIV_SPEC_REVISION,
            &ELFCompactAttrParser::integerAttribute,
        },
        {
            CapstoneAttrs::STACK_ALIGN,
            &CapstoneAttributeParser::stackAlign,
        },
        {
            CapstoneAttrs::UNALIGNED_ACCESS,
            &CapstoneAttributeParser::unalignedAccess,
        },
        {
            CapstoneAttrs::ATOMIC_ABI,
            &CapstoneAttributeParser::atomicAbi,
        },
};

Error CapstoneAttributeParser::atomicAbi(unsigned Tag) {
  uint64_t Value = de.getULEB128(cursor);
  printAttribute(Tag, Value, "Atomic ABI is " + utostr(Value));
  return Error::success();
}

Error CapstoneAttributeParser::unalignedAccess(unsigned tag) {
  static const char *const strings[] = {"No unaligned access",
                                        "Unaligned access"};
  return parseStringAttribute("Unaligned_access", tag, ArrayRef(strings));
}

Error CapstoneAttributeParser::stackAlign(unsigned tag) {
  uint64_t value = de.getULEB128(cursor);
  std::string description =
      "Stack alignment is " + utostr(value) + std::string("-bytes");
  printAttribute(tag, value, description);
  return Error::success();
}

Error CapstoneAttributeParser::handler(uint64_t tag, bool &handled) {
  handled = false;
  for (const auto &AH : displayRoutines) {
    if (uint64_t(AH.attribute) == tag) {
      if (Error e = (this->*AH.routine)(tag))
        return e;
      handled = true;
      break;
    }
  }

  return Error::success();
}
