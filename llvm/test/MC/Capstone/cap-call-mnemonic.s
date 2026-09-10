# The `call` mnemonic has THREE definitions, and all of them must assemble.
# Capstone issues C-38 (register form) and C-45 (register+symbol form), both FIXED.
#
#   CAP_CALL       `call $rd, $rs1`   two capability registers (domain crossing)
#   PseudoCALLReg  `call $rd, $func`  register + symbol
#   PseudoCALL     `call $func`       symbol only  (pinned by cap-call-symbol.s)
#
# C-38: parseCallSymbol claimed ANY identifier in the last operand position as a
# symbol, so `call a0, a1` was read as a call to a symbol named "a1" and CAP_CALL
# was unreachable -- while the DISASSEMBLER prints the CAP_CALL encoding as exactly
# `call a0, a1`. Fixed by declining register names there.
#
# C-45: validateTargetOperandClass coerces an integer register to its capability
# form IN PLACE, and the matcher does not restore it when the candidate then fails.
# CAP_CALL sorts before PseudoCALLReg (both take two operands and tie on count), so
# the failed CAP_CALL trial left `a0` rewritten as C10 and PseudoCALLReg then saw a
# capability register where it wanted an integer one. Fixed by making the coercion
# idempotent with a reverse arm. `call a0, foo` and the real codegen shape
# `call t0, __riscv_save_12` -- emitted for spill libcalls and by the machine
# outliner -- could not be assembled at all before that, so -S output was not
# reassemblable.
#
# MUTATION: replace `call a0, a1` with `capenter a0, a1` (a sibling CAP_OP that also
# round-trips) -> it assembles to a DIFFERENT encoding and the check fails, which
# shows these checks are bound to these instructions and not merely to "something
# assembled".
#
# RUN: llvm-mc -triple capstone64 -show-encoding %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple capstone64 -filetype=obj %s | llvm-objdump -M no-aliases -d -r - | FileCheck %s --check-prefix=OBJ

# ---------------------------------------------------------------------------
# THE PIN (C-45). A register/register `call` must stay CAP_CALL and must NEVER
# become a call to a symbol that happens to be spelled like a register.
#
# This is the guard the suite could not previously provide. On a target where the
# operand is not restored to its integer form, `call a0, a1` matches PseudoCALLReg
# instead and emits a PLT call against an undefined symbol literally named `a1` --
# silently different object code from the same text the disassembler prints. The
# encoding check plus OBJ-NOT below fail loudly if that ever happens here.
# ---------------------------------------------------------------------------

# ASM: call a0, a1
# ASM-SAME: encoding: [0x5b,0x95,0x05,0x40]
# OBJ: 5b 95 05 40 call a0, a1
call a0, a1

# The codegen form PseudoDomCall expands to, rd == rs1.
# ASM: call a0, a0
# ASM-SAME: encoding: [0x5b,0x15,0x05,0x40]
# OBJ: 5b 15 05 40 call a0, a0
call a0, a0

# The cN spelling names the same registers as aN (one register file), so this is
# the same instruction and the same encoding.
# ASM: call a0, a1
# ASM-SAME: encoding: [0x5b,0x95,0x05,0x40]
# OBJ: 5b 95 05 40 call a0, a1
call c10, c11

# ---------------------------------------------------------------------------
# C-45: the register+symbol form reaches PseudoCALLReg and expands to auipc+jalr
# through the named register, with a call relocation.
# ---------------------------------------------------------------------------

# ASM: call a0, foo
# OBJ: auipc a0
# OBJ-NEXT: R_Capstone_CALL_PLT foo
# OBJ: jalr a0
call a0, foo

# The shape codegen actually emits: spill libcalls via t0, and the machine outliner.
# ASM: call t0, __riscv_save_12
# OBJ: auipc t0
# OBJ-NEXT: R_Capstone_CALL_PLT __riscv_save_12
# OBJ: jalr t0
call t0, __riscv_save_12

# The three register/register calls above must produce NO relocation. If one ever
# degrades into a symbol call, a third R_Capstone_CALL_PLT appears and this fails.
# OBJ-NOT: R_Capstone_CALL_PLT
