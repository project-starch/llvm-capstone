//===-- CapstoneFixupKinds.h - Capstone Specific Fixup Entries -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_MCTARGETDESC_CapstoneFIXUPKINDS_H
#define LLVM_LIB_TARGET_Capstone_MCTARGETDESC_CapstoneFIXUPKINDS_H

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCFixup.h"
#include <utility>

#undef Capstone

namespace llvm::Capstone {
enum Fixups {
  // 20-bit fixup corresponding to %hi(foo) for instructions like lui
  fixup_capstone_hi20 = FirstTargetFixupKind,
  // 12-bit fixup corresponding to %lo(foo) for instructions like addi
  fixup_capstone_lo12_i,
  // 12-bit fixup corresponding to foo-bar for instructions like addi
  fixup_capstone_12_i,
  // 12-bit fixup corresponding to %lo(foo) for the S-type store instructions
  fixup_capstone_lo12_s,
  // 20-bit fixup corresponding to %pcrel_hi(foo) for instructions like auipc
  fixup_capstone_pcrel_hi20,
  // 12-bit fixup corresponding to %pcrel_lo(foo) for instructions like addi
  fixup_capstone_pcrel_lo12_i,
  // 12-bit fixup corresponding to %pcrel_lo(foo) for the S-type store
  // instructions
  fixup_capstone_pcrel_lo12_s,
  // 20-bit fixup for symbol references in the jal instruction
  fixup_capstone_jal,
  // 12-bit fixup for symbol references in the branch instructions
  fixup_capstone_branch,
  // 11-bit fixup for symbol references in the compressed jump instruction
  fixup_capstone_rvc_jump,
  // 8-bit fixup for symbol references in the compressed branch instruction
  fixup_capstone_rvc_branch,
  // 6-bit fixup for symbol references in instructions like c.li
  fixup_capstone_rvc_imm,
  // Fixup representing a legacy no-pic function call attached to the auipc
  // instruction in a pair composed of adjacent auipc+jalr instructions.
  fixup_capstone_call,
  // Fixup representing a function call attached to the auipc instruction in a
  // pair composed of adjacent auipc+jalr instructions.
  fixup_capstone_call_plt,

  // Qualcomm specific fixups
  // 12-bit fixup for symbol references in the 48-bit Xqcibi branch immediate
  // instructions
  fixup_capstone_qc_e_branch,
  // 32-bit fixup for symbol references in the 48-bit qc.e.li instruction
  fixup_capstone_qc_e_32,
  // 20-bit fixup for symbol references in the 32-bit qc.li instruction
  fixup_capstone_qc_abs20_u,
  // 32-bit fixup for symbol references in the 48-bit qc.j/qc.jal instructions
  fixup_capstone_qc_e_call_plt,

  // Andes specific fixups
  // 10-bit fixup for symbol references in the xandesperf branch instruction
  fixup_capstone_nds_branch_10,

  // Used as a sentinel, must be the last
  fixup_capstone_invalid,
  NumTargetFixupKinds = fixup_capstone_invalid - FirstTargetFixupKind
};
} // end namespace llvm::Capstone

#endif
