# The domain-entry glue hand-encodes SPLIT (funct7 0b0000110 under the
# capability opcode 0x5b) with a `.insn r` directive, because the backend
# defines no SPLIT instruction.  Pin that the directive assembles to the bytes
# the glue relies on, and that the disassembler has NO accidental decode for
# them: objdump must print `<unknown>` on exactly those bytes.
# (cap-invalid-encodings.txt pins the same word through llvm-mc's decoder.)
# Measured 2026-09-04 on the branch tools.
#
# MUTATION: change funct7 6 to 2 on the first line -> it assembles as
# [0x5b,0x95,0xc5,0x04] and objdump decodes it as `tighten a0, a1, 0xc`, so
# the OBJ line for those bytes fails (performed 2026-09-04).
#
# RUN: llvm-mc -triple capstone64 -show-encoding %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple capstone64 -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d %t.o | FileCheck %s --check-prefix=OBJ

.insn r 0x5b, 1, 6, a0, a1, a2
# ASM: .insn r 91, 1, 6, a0, a1, a2 # encoding: [0x5b,0x95,0xc5,0x0c]
# OBJ: 5b 95 c5 0c <unknown>
.insn r 0x5b, 1, 6, s0, s1, s2
# ASM: .insn r 91, 1, 6, s0, s1, s2 # encoding: [0x5b,0x94,0x24,0x0d]
# OBJ: 5b 94 24 0d <unknown>
