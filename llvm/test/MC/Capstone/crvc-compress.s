# cap-crvc: the assembler compresses each 32-bit capability op that fits a CRVC
# form, leaves the rest 4-byte, and never emits the standard forms the CRVC mode
# displaces (here C.FLD and C.FLDSP, which +cap-base-rvc emits without it).
# RUN: llvm-mc -triple=capstone64 -mattr=+c,+d,+cap-base-rvc,+cap-crvc -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d --mattr=+c,+d,+cap-base-rvc,+cap-crvc %t.o | FileCheck %s
# RUN: llvm-mc -triple=capstone64 -mattr=+c,+d,+cap-base-rvc -filetype=obj %s -o %t.off.o
# RUN: llvm-objdump -d --mattr=+c,+d,+cap-base-rvc %t.off.o | FileCheck %s --check-prefix=OFF

# CHECK: 42 25 c.ldcsp a0, 0x10(sp)
# OFF:   5b 35 01 01 ldc a0, 0x10(sp)
ldc c10, 16(c2)
# CHECK: a6 bf c.stcsp s1, 0x3f0(sp)
stc c9, 1008(c2)
# CHECK: 80 30 c.ldc s0, 0x20(s1)
ldc c8, 32(c9)
# CHECK: 1c a0 c.stc a5, 0x0(s0)
stc c15, 0(c8)
# CHECK: 39 70 c.cincoffset16sp sp, -0x40
cincoffsetimm c2, c2, -64
# CHECK: 80 90 c.movc s0, s1
# OFF:   5b 94 04 14 movc s0, s1
movc c8, c9
# CHECK: 88 95 c.scc a0, gp, a1
scc c10, c3, x11
# CHECK: 14 9a c.cincoffset a2, a3
cincoffset c12, c12, x13
# Outside the forms: a register outside c8-c15, an offset that is not a
# multiple of 16.
# CHECK: 5b 14 0a 14 movc s0, s4
movc c8, c20
# CHECK: 5b 35 81 00 ldc a0, 0x8(sp)
ldc c10, 8(c2)
# The displaced FP forms stay 4-byte under the feature.
# CHECK: 07 34 84 00 fld fs0, 0x8(s0)
# OFF:   00 24 fld fs0, 0x8(s0)
fld f8, 8(c8)
# CHECK: 87 30 01 01 fld ft1, 0x10(sp)
# OFF:   c2 20 fld ft1, 0x10(sp)
fld f1, 16(c2)
