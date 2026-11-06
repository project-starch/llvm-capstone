# SHRINKTO: OPC_CAP_OP (0b1011011), funct3 000, I-type, signed 12-bit length.
# RUN: llvm-mc -triple=capstone64 -show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple=capstone64 %s --defsym=BAD=1 2>&1 | FileCheck %s --check-prefix=ERR

# CHECK: shrinkto a0, a1, 16 # encoding: [0x5b,0x85,0x05,0x01]
shrinkto a0, a1, 16
# CHECK: shrinkto a4, a4, 2047 # encoding: [0x5b,0x07,0xf7,0x7f]
shrinkto a4, a4, 2047
.ifdef BAD
# ERR: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
shrinkto a0, a1, 2048
.endif
