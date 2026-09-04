# Register spellings.  A capability operand is a C register (C0..C31) that
# shares its number with the X register it extends; the assembler ACCEPTS the
# `cN` spelling and the `xN` spelling, and the printer emits the ABI name for
# both -- there is no `c`-name on output, by design (CapstoneRegisterInfo.td:
# "the assembly must keep printing a0, not c10").  Measured 2026-09-04.
#
# MUTATION: change the expected `movc a0, a1` for the c-spelled line to
# `movc c10, c11` -> the CHECK fails, which shows the printer never emits
# c-names (performed 2026-09-04).
#
# RUN: llvm-mc -triple capstone64 -show-encoding %s | FileCheck %s

# CHECK: movc a0, a1  # encoding: [0x5b,0x95,0x05,0x14]
movc c10, c11
# CHECK: movc a0, a1  # encoding: [0x5b,0x95,0x05,0x14]
movc x10, x11
# CHECK: movc a0, a1  # encoding: [0x5b,0x95,0x05,0x14]
movc a0, a1
# CHECK: ldc a0, 8(a1)  # encoding: [0x5b,0xb5,0x85,0x00]
ldc c10, 8(c11)
# CHECK: cincoffset a0, a1, a2  # encoding: [0x5b,0x95,0xc5,0x18]
cincoffset c10, c11, x12
