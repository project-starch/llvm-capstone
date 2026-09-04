# Operand-diagnostic tests for every Capstone capability instruction class:
# immediates one past each end of their range, wrong arity, a non-capability
# register where a capability is required, a symbol modifier where an
# immediate is required, an unknown CSR name, and the register the encoding
# forbids (mrev with rd = zero, GPCRNoC0), and one extra operand on every
# fixed-arity instruction.  Each diagnostic is anchored to its
# line and column, so a changed operand class moves the message and this
# file notices.  Measured 2026-09-04 on the branch llvm-mc.
#
# MUTATION: change `2048` on the first line to `2047` -> that line assembles,
# the anchored diagnostic is not produced, and FileCheck fails on it
# (performed 2026-09-04).
#
# RUN: not llvm-mc -triple capstone64 < %s 2>&1 | FileCheck %s

ldc a0, 2048(a1) # CHECK: :[[@LINE]]:9: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
stc a0, -2049(a1) # CHECK: :[[@LINE]]:9: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
cincoffsetimm a0, a1, 2048 # CHECK: :[[@LINE]]:23: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
cincoffsetimm a0, a1, -2049 # CHECK: :[[@LINE]]:23: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
lcc a0, a1, 32 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
lcc a0, a1, -1 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
tighten a0, a1, 32 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
ccsrrw a0, 4096, a1 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]
cjalr ra, 2048(a0) # CHECK: :[[@LINE]]:11: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
delin a0, a1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
movc a0 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
mrev zero, a0 # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
ldc fa0, 0(a1) # CHECK: :[[@LINE]]:5: error: invalid operand for instruction
lcc a0, a1, %lo(x) # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
ccsrrw a0, nosuchcsr, a1 # CHECK: :[[@LINE]]:12: error: operand must be a valid system register name or an integer in the range [0, 4095]
shrink a0, a1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
seal a0 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
cincoffset a0, a1, 5 # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
scc a0, a1, 5 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
init a0, a1, a2, a3 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
drop a0, a1 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
revoke a0, a1 # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
call a0, a1, a2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
capenter a0, a1, a2 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
return a0, a1, a2, a3 # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
return a0, a1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# capexit was a compiler-only phantom (C-36); the mnemonic itself is gone.
capexit a0, a1 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
