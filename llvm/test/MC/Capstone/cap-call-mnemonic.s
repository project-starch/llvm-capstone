# The register form of CAP_CALL round-trips through the MC layer.
# Capstone issue C-38 (the register-form CAP_CALL mnemonic collision), FIXED.
#
# The bug (measured 2026-09-04): assembling `call a0, a1` failed with "invalid
# operand for instruction", because three defs share the "call" mnemonic --
# PseudoCALL (`call $func`), PseudoCALLReg (`call $rd, $func`) and CAP_CALL
# (`call $rd, $rs1`, two capability registers) -- and parseCallSymbol claimed any
# identifier in the last operand position as a symbol, so `a1` became a symbol
# named "a1" and CAP_CALL was unreachable. Meanwhile the DISASSEMBLER prints the
# CAP_CALL encoding 0x5b 0x95 0x05 0x40 as exactly `call a0, a1`, so for this one
# instruction object -> text -> object was impossible by construction.
#
# The fix (2026-09-09) is parser precedence, not a rename: parseCallSymbol now
# declines register names, so the operand falls through to register parsing and
# CAP_CALL matches. The mnemonic and the disassembly text are unchanged, which is
# what makes the round-trip close.
#
# cap-call-symbol.s is the control: `call foo` must keep assembling to the
# PseudoCALL expansion, so a fix that breaks the pseudo cannot pass. cap-invalid.s
# pins that `call a0, a1, a2` still errors.
#
# MUTATION: replace `call a0, a1` below with `capenter a0, a1` (a sibling CAP_OP
# instruction that also round-trips) -> the encoding CHECK fails, which shows the
# check is bound to this instruction and not merely to "something assembled".
#
# RUN: llvm-mc -triple capstone64 -show-encoding %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple capstone64 -filetype=obj %s | llvm-objdump -M no-aliases -d - | FileCheck %s --check-prefix=OBJ

# The symbol path is bounded from the other side by cap-call-symbol.s (`call foo`
# -> PseudoCALL), which this fix leaves untouched.
#
# NOT COVERED HERE, and NOT a regression of this fix: the register+symbol form
# `call a0, foo` (PseudoCALLReg) does not assemble on this target. Measured on the
# PRE-FIX binary (the 2026-09-04 main-checkout llvm-mc), which rejects all three of
# `call a0, a1`, `call a0, a0` and `call a0, foo` identically -- so that form never
# worked and this change neither fixes nor breaks it. Recorded as a separate gap
# rather than folded in here; C-38 is the CAP_CALL collision only.

# ASM: call a0, a1
# ASM-SAME: encoding: [0x5b,0x95,0x05,0x40]
# OBJ: 5b 95 05 40 call a0, a1
call a0, a1

# The codegen form PseudoDomCall expands to (rd == rs1, both a0).
# ASM: call a0, a0
# ASM-SAME: encoding: [0x5b,0x15,0x05,0x40]
# OBJ: 5b 15 05 40 call a0, a0
call a0, a0
