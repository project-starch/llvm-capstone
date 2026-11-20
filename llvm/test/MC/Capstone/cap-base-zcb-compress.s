# cap-base-zcb: Zcb's byte and halfword loads and stores compress with a
# capability base in c8-c15 (stage-1 style: standard encodings, codegen-only
# forms). Out of range, or a register outside x8-x15/c8-c15, stays 4-byte. Plain
# +zcb names integer bases only, so it compresses none of these (control).
# RUN: llvm-mc -triple=capstone64 -mattr=+c,+cap-base-rvc,+cap-base-zcb -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d --mattr=+c,+zcb %t.o | FileCheck %s
# RUN: llvm-mc -triple=capstone64 -mattr=+c,+cap-base-rvc,+zcb -filetype=obj %s -o %t.off.o
# RUN: llvm-objdump -d --mattr=+c,+zcb %t.off.o | FileCheck %s --check-prefix=OFF

# CHECK: e8 81 lbu a0, 0x3(a1)
# OFF:   03 c5 35 00 lbu a0, 0x3(a1)
lbu x10, 3(c11)
# CHECK: a0 84 lhu s0, 0x2(s1)
lhu x8, 2(c9)
# CHECK: d0 87 lh a2, 0x0(a5)
lh x12, 0(c15)
# CHECK: 54 88 sb a3, 0x1(s0)
# OFF:   a3 00 d4 00 sb a3, 0x1(s0)
sb x13, 1(c8)
# CHECK: 38 8d sh a4, 0x2(a0)
sh x14, 2(c10)
# CHECK: 03 c5 45 00 lbu a0, 0x4(a1)
lbu x10, 4(c11)
# CHECK: 03 ca 05 00 lbu s4, 0x0(a1)
lbu x20, 0(c11)
# CHECK: 03 45 0a 00 lbu a0, 0x0(s4)
lbu x10, 0(c20)
