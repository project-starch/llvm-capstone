# Compressed capability instructions (cap-crvc), stage 2 option B: encodings in
# RV128's C.LQSP/C.SQSP/C.LQ/C.SQ layouts, C.ADDI16SP's immediate in C.LUI x0's
# slot, and three Zcb-reserved Q0 100 slots. Each immediate at both ends.
# Without the feature the same bytes read as the standard forms they displace.
# plans/2026-11-21-capability-rvc-stage2-prototype.md
# RUN: llvm-mc -triple=capstone64 -mattr=+cap-crvc -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=capstone64 -mattr=+cap-crvc,+d -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d --mattr=+cap-crvc,+d %t.o | FileCheck %s --check-prefix=DIS
# RUN: llvm-objdump -d --mattr=+c,+d %t.o | FileCheck %s --check-prefix=STD
# RUN: llvm-readelf -h %t.o | FileCheck %s --check-prefix=FLAGS
# RUN: not llvm-mc -triple=capstone64 %s 2>&1 | FileCheck %s --check-prefix=NOFEAT

# NOFEAT: error: instruction requires the following: Compressed capability instructions

# FLAGS: Flags: 0x105

# CHECK: c.ldcsp a0, 1008(sp)    # encoding: [0x7e,0x35]
# DIS: 7e 35 c.ldcsp a0, 0x3f0(sp)
# STD: 7e 35 fld fa0, 0x1f8(sp)
c.ldcsp c10, 1008(c2)
# CHECK: c.ldcsp ra, 0(sp)       # encoding: [0x82,0x20]
# DIS: 82 20 c.ldcsp ra, 0x0(sp)
c.ldcsp c1, 0(c2)
# CHECK: c.stcsp a0, 1008(sp)    # encoding: [0xaa,0xbf]
# DIS: aa bf c.stcsp a0, 0x3f0(sp)
# STD: aa bf fsd fa0, 0x1f8(sp)
c.stcsp c10, 1008(c2)
# CHECK: c.stcsp zero, 16(sp)    # encoding: [0x02,0xa8]
c.stcsp c0, 16(c2)
# CHECK: c.ldc s0, 496(a5)       # encoding: [0xe0,0x3f]
# DIS: e0 3f c.ldc s0, 0x1f0(a5)
# STD: e0 3f fld fs0, 0xf8(a5)
c.ldc c8, 496(c15)
# CHECK: c.ldc s1, 0(s0)         # encoding: [0x04,0x20]
c.ldc c9, 0(c8)
# CHECK: c.stc a5, 256(s0)       # encoding: [0x1c,0xa4]
# DIS: 1c a4 c.stc a5, 0x100(s0)
# STD: 1c a4 fsd fa5, 0x8(s0)
c.stc c15, 256(c8)
# CHECK: c.cincoffset16sp sp, -512  # encoding: [0x01,0x70]
# DIS: 01 70 c.cincoffset16sp sp, -0x200
# STD: 01 70 c.lui zero, 0xfffe0
c.cincoffset16sp c2, -512
# CHECK: c.cincoffset16sp sp, 496   # encoding: [0x7d,0x60]
c.cincoffset16sp c2, 496
# CHECK: c.movc s0, a5           # encoding: [0x80,0x93]
# DIS: 80 93 c.movc s0, a5
# STD: 80 93 <unknown>
c.movc c8, c15
# CHECK: c.scc s1, gp, a0        # encoding: [0x04,0x95]
# DIS: 04 95 c.scc s1, gp, a0
# STD: 04 95 <unknown>
c.scc c9, c3, x10
# CHECK: c.cincoffset a1, a2     # encoding: [0x90,0x99]
# DIS: 90 99 c.cincoffset a1, a2
# STD: 90 99 <unknown>
c.cincoffset c11, x12
