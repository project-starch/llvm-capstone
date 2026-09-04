# The register form of CAP_CALL does not round-trip through the MC layer.
# Capstone issue C-38 (the register-form CAP_CALL mnemonic collision).
#
# Measured 2026-09-04 on the current llvm-mc: assembling `call a0, a1` fails
# with "invalid operand for instruction" at column 10, because the mnemonic
# collides with PseudoCALL's `call $func` (CapstoneInstrInfo.td, call_symbol
# parsed by parseCallSymbol) and the symbol parser wins.  Disassembling the
# CAP_CALL encoding 0x5b 0x95 0x05 0x40 prints exactly `call a0, a1` -- text
# the assembler will not accept.  So for this one instruction, object -> text
# -> object is impossible by construction.  The fix (a distinct mnemonic, or
# parser precedence) is a backend decision; this file is XFAIL until it lands
# and reports XPASS the moment it does.
#
# cap-call-symbol.s is the control: `call foo` must keep assembling to the
# PseudoCALL expansion, so a fix that breaks the pseudo cannot pass.
#
# MUTATION: replace `call a0, a1` below with `capenter a0, a1` (a sibling
# CAP_OP instruction that does round-trip) -> the assembler arm passes and the
# encoding CHECK fails, which shows the check is bound to this instruction.
#
# RUN: llvm-mc -triple capstone64 -show-encoding %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple capstone64 -filetype=obj %s | llvm-objdump -M no-aliases -d - | FileCheck %s --check-prefix=OBJ
# XFAIL: *

# ASM: call a0, a1
# ASM-SAME: encoding: [0x5b,0x95,0x05,0x40]
# OBJ: 5b 95 05 40 {{[[:space:]]+}}call a0, a1
call a0, a1
