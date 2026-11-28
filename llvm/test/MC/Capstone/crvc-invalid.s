# cap-crvc: operands outside each form's range do not assemble to it.
# RUN: not llvm-mc -triple=capstone64 -mattr=+cap-crvc %s 2>&1 | FileCheck %s

c.ldcsp c0, 16(c2)
# CHECK: :[[@LINE-1]]:9: error: invalid operand for instruction
c.ldcsp c10, 8(c2)
# CHECK: :[[@LINE-1]]:14: error: immediate must be a multiple of 16 bytes in the range [0, 1008]
c.ldcsp c10, 1024(c2)
# CHECK: :[[@LINE-1]]:14: error: immediate must be a multiple of 16 bytes in the range [0, 1008]
c.ldc c7, 16(c8)
# CHECK: :[[@LINE-1]]:7: error: invalid operand for instruction
c.ldc c8, 512(c9)
# CHECK: :[[@LINE-1]]:11: error: immediate must be a multiple of 16 bytes in the range [0, 496]
c.cincoffset16sp c2, 0
# CHECK: :[[@LINE-1]]:22: error: immediate must be a multiple of 16 bytes and non-zero in the range [-512, 496]
c.movc c16, c8
# CHECK: :[[@LINE-1]]:8: error: invalid operand for instruction
c.scc c8, c4, x9
# CHECK: :[[@LINE-1]]:11: error: invalid operand for instruction
