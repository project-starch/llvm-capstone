# See LICENSE for license details.

#*****************************************************************************
# sublet-split-shares-node.S  (a non-linear SPLIT keeps its node, so neither half outlives its lifetime; 2026-10-13)
#-----------------------------------------------------------------------------
#
# P is a NONLIN parent with MANAGE; C = derive(P, 1024) at P's base; C is split at +512 into C (the lower half,
# rs1 rewritten) and C2 (the upper half, rd). Before this change C2 got a sibling node at C's depth, inserted
# right after C's node: crevoke(P, C) then ended C's node and its deeper run, and C2 survived on the sibling (the
# "escape"). Now both halves share C's node.
#
#   #1  a load through C after the split                                                       0x1111
#   #2  a load through C2                                                                      0x2222
#   #3  mcause of crevoke(P, C)                                                                0
#   #4  mcause of a load through C2                                 new 25 (0x19); old 0 (C2 survived the kill)
#   #5  mcause of a load through C                                                             25
#   #6  the same the other way round: C3 = derive(P, 1024) at +1024, C4 = split(C3, +1536);
#       mcause of crevoke(P, C4): C4 shares C3's node, a direct child                          0
#       (old: 29 -- the sibling node was created LINEAR by init_req, so the kill refused it as not a non-linear
#        child; the refusal is itself the kill's linearity check firing)
#   #7  mcause of a load through C3                                 new 25 (the shared node died); old 0
#
# Instruction names are lower case in comments because this file goes through cpp.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li s9, 0
lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

# a1: NONLIN RW + MANAGE over [buf, buf_end)
CAPCREATE(a1)
li a0, CAP_TYPE_NONLIN
lla a3, buf
lla a4, buf_end
li a5, 14
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)

# C = derive(a1, 1024) at buf
li t2, 1024
la s10, r1
CDERIVE(s1, a1, t2)
r1:
# split C at buf+512: s1 = [buf, buf+512), s2 = [buf+512, buf+1024)
lla a3, buf
li t2, 512
add a3, a3, t2
la s10, r2
SPLIT(s2, s1, a3)
r2:
li t3, 0x1111
sd t3, 0(s1)
ld t4, 0(s1)
CAPPRINT(t4)                   # #1: 0x1111
li t3, 0x2222
sd t3, 0(s2)
ld t4, 0(s2)
CAPPRINT(t4)                   # #2: 0x2222

li s9, 0
la s10, r3
CREVOKE(a1, s1)
r3:
CAPPRINT(s9)                   # #3: 0

li s9, 0
la s10, r4
ld t4, 0(s2)
r4:
CAPPRINT(s9)                   # #4: new 25, old 0

li s9, 0
la s10, r5
ld t4, 0(s1)
r5:
CAPPRINT(s9)                   # #5: 25

# C3 = derive(a2, 1024) at buf+1024, split at buf+1536, crevoke the UPPER half
li t2, 1024
CINCOFFSET(a2, a1, t2)
la s10, r6
CDERIVE(s3, a2, t2)
r6:
lla a3, buf
li t2, 1536
add a3, a3, t2
la s10, r7
SPLIT(s4, s3, a3)
r7:
li s9, 0
la s10, r8
CREVOKE(a1, s4)
r8:
CAPPRINT(s9)                   # #6: 0

li s9, 0
la s10, r9
ld t4, 0(s3)
r9:
CAPPRINT(s9)                   # #7: new 25, old 0

CAP_PASS(s11)

.align 4
trap_handler:
  csrr s9, mcause
  csrw mepc, s10
  mret

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
buf:
  .zero 4096
buf_end:
  TEST_DATA
RVTEST_DATA_END
