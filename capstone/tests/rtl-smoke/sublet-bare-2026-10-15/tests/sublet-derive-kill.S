# See LICENSE for license details.

#*****************************************************************************
# sublet-derive-kill.S  (Sublet child lifetimes on the RTL: CDERIVE / CREVOKE / MANAGE, 2026-10-13)
#-----------------------------------------------------------------------------
#
# The first arm of the capstone-sublet instruction work. Every reading is a CAPPRINT; traps are recorded by the
# handler into s9 (mcause) and resume at the address the test left in s10 before the instruction.
#
#   #1  child perm: a NONLIN RW+MANAGE parent derives a child with R/W plus MANAGE              14 (0xe)
#   #2  child start - buf                                                                        256
#   #3  child end - buf                                                                          768
#   #4  child type field (type - 1): NONLIN                                                      1
#   #5  a store through the child, loaded back through it                                       0x1234
#   #6  a grandchild's perm (derive from the child)                                              14
#   #7  mcause of crevoke(parent, grandchild): not a DIRECT child                                29 (0x1d)
#   #8  mcause of crevoke(child without MANAGE, grandchild): MANAGE needed                       27 (0x1b)
#   #9  mcause after crevoke(child, grandchild): succeeds                                        0
#   #10 mcause of a load through the dead grandchild                                             25 (0x19)
#   #11 mcause of crevoke(child, grandchild) again: the child's reference is stale               25
#   #12 mcause of a derive with size 0                                                           29
#   #13 mcause of a derive whose window leaves the parent                                        29
#   #14 mcause after crevoke(parent-copy, child): a CINCOFFSET copy shares the node, succeeds     0
#   #15 mcause of a load through the dead child                                                  25
#   #16 a fresh derive after the kills, perm (the freed slots are reused)                        14
#   #17 a load through the fresh child                                                           0x5678
#   #18 mcause of a derive from a LINEAR parent                                                  26 (0x1a)
#   #19 mcause of a derive whose size operand is a capability                                    24 (0x18)
#
# Positive control: on a tree without the instructions (capstone-sublet before this commit) every derive and every
# crevoke raises 2 (illegal instruction), so #1 reads the handler's record and nothing else matches.
# The instruction names are written in lower case in comments because this file goes through cpp.

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

# a1: NONLIN RW + MANAGE over [buf, buf_end) (CAPPERM takes the four rights bits; 14 = RW with MANAGE)
CAPCREATE(a1)
li a0, CAP_TYPE_NONLIN
lla a3, buf
lla a4, buf_end
li a5, 14
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)

# a2: the parent's cursor moved to buf+256 (same node)
li t2, 256
CINCOFFSET(a2, a1, t2)

# s1 = derive(a2, 512): [buf+256, buf+768)
li t2, 512
la s10, r1
CDERIVE(s1, a2, t2)
r1:
LCC(t1, s1, x5)
CAPPRINT(t1)                   # #1: 14
LCC(t1, s1, x3)
lla a3, buf
sub t1, t1, a3
CAPPRINT(t1)                   # #2: 256
LCC(t1, s1, x4)
sub t1, t1, a3
CAPPRINT(t1)                   # #3: 768
LCC(t1, s1, x1)
CAPPRINT(t1)                   # #4: 1

li t3, 0x1234
sd t3, 0(s1)
ld t4, 0(s1)
CAPPRINT(t4)                   # #5: 0x1234

# s2 = derive(s1, 64): the grandchild
li t3, 64
la s10, r2
CDERIVE(s2, s1, t3)
r2:
LCC(t1, s2, x5)
CAPPRINT(t1)                   # #6: 14

# crevoke(a1, s2): s2 is not a direct child of a1's node
li s9, 0
la s10, r3
CREVOKE(a1, s2)
r3:
CAPPRINT(s9)                   # #7: 29

# s3 = tighten(s1, 6): the child without MANAGE (rights 6 = R/W; 7 would ask for X, which the child lacks: 29)
la s10, r3b
TIGHTEN(s3, s1, x6)
r3b:
li s9, 0
la s10, r4
CREVOKE(s3, s2)
r4:
CAPPRINT(s9)                   # #8: 27

# crevoke(s1, s2): ends the grandchild
li s9, 0
la s10, r5
CREVOKE(s1, s2)
r5:
CAPPRINT(s9)                   # #9: 0

li s9, 0
la s10, r6
ld t4, 0(s2)
r6:
CAPPRINT(s9)                   # #10: 25

li s9, 0
la s10, r7
CREVOKE(s1, s2)
r7:
CAPPRINT(s9)                   # #11: 25

# derive with size 0
li t3, 0
li s9, 0
la s10, r8
CDERIVE(s4, s1, t3)
r8:
CAPPRINT(s9)                   # #12: 29

# derive whose window leaves the parent: 4096 from buf+256
li t3, 4096
li s9, 0
la s10, r9
CDERIVE(s4, a2, t3)
r9:
CAPPRINT(s9)                   # #13: 29

# crevoke(a2, s1): a2 shares a1's node, s1 is its direct child
li s9, 0
la s10, r10
CREVOKE(a2, s1)
r10:
CAPPRINT(s9)                   # #14: 0

li s9, 0
la s10, r11
ld t4, 0(s1)
r11:
CAPPRINT(s9)                   # #15: 25

# a fresh derive after the kills: [buf, buf+128)
li t3, 128
la s10, r12
CDERIVE(s5, a1, t3)
r12:
LCC(t1, s5, x5)
CAPPRINT(t1)                   # #16: 14
li t3, 0x5678
sd t3, 0(s5)
ld t4, 0(s5)
CAPPRINT(t4)                   # #17: 0x5678

# a LINEAR parent: derive refuses with 26
CAPCREATE(s6)
li a0, CAP_TYPE_LIN
lla a3, buf2
lla a4, buf2_end
li a5, CAP_PERM_RW
CAPTYPE(s6, a0)
CAPBOUND(s6, a3, a4)
CAPPERM(s6, a5)
li t3, 64
li s9, 0
la s10, r13
CDERIVE(s7, s6, t3)
r13:
CAPPRINT(s9)                   # #18: 26

# a capability as the size operand: 24
li s9, 0
la s10, r14
CDERIVE(s7, a1, a1)
r14:
CAPPRINT(s9)                   # #19: 24

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
.align 6
buf2:
  .zero 256
buf2_end:
  TEST_DATA
RVTEST_DATA_END
