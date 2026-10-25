# See LICENSE for license details.

#*****************************************************************************
# sublet-mrev-orphan.S  (the claim-auditor's counterexample to "MREV deepening is unreachable", 2026-10-13)
#-----------------------------------------------------------------------------
#
# On this RTL a LINEAR capability can be duplicated (R-21: TIGHTEN and SHRINKTO return rs1 unchanged beside a LINEAR
# rd), so one node can carry a LINEAR capability, which MREV accepts, and a NONLIN one, which CDERIVE accepts. Then:
#   L on node N (depth d); L2 = tighten(L, same rights)   -- a LINEAR duplicate on N (R-21)
#   delin(L2)                                              -- N.linear = 0, L2 NONLIN, L still LINEAR
#   C = derive(L2, 256)                                    -- child right after N at depth d + 1
#   H1 = mrev(L); H2 = mrev(L)                             -- list H1(d) -> H2(d+1) -> N(d+2) -> C(d+1)
#   revoke(H2)                                             -- bound d + 1: N is ended, C (depth d + 1) stops the walk
# Without the subtree deepening, C outlives its parent: a load through C reads 0 (QEMU's rule would deepen C to d + 2
# at the second MREV, so the walk ends it and the load reads 25).
#
#   #1  mcause after tighten: 0 (the duplicate exists: LCC type of L reads 1 - 1 = 0, i.e. still LINEAR)      0
#   #2  type field of L after tighten (type - 1): LINEAR = 0; on a tree where TIGHTEN moves, L is cnull       0
#   #3  mcause of the first mrev                                                                               0
#   #4  mcause of the second mrev                                                                              0
#   #5  mcause of revoke(H2)                                                                                   0
#   #6  mcause of a load through C: WITH deepening 25 (0x19); without it 0 (the orphan)
#   #7  mcause of a load through L2: 25 on both (N was ended)
#
# The split variant (the auditor's second escape): S = split(L, mid) puts a sibling S(d) right after N, before C;
# mrev(L) once gives H1(d) -> N(d+1) -> S(d) -> C(d+1); revoke(H1) ends N and stops at S, so C survives even with
# deepening. QEMU inserts siblings BEFORE the node (dup_before), which keeps a node's children contiguous behind it.
#   #8  mcause of a load through C2 (the child of the split's parent) after revoke(H3):
#       with QEMU's sibling placement 25; with the sibling after the node 0
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

# L: LINEAR RW over [buf, buf+4096) on node 2
CAPCREATE(s2)
li a0, CAP_TYPE_LIN
lla a2, buf
lla a3, buf_end
li a4, CAP_PERM_RW
CAPTYPE(s2, a0)
CAPBOUND(s2, a2, a3)
CAPPERM(s2, a4)

# the verdict capability s11 and the second section's L7 are split siblings of node 2, re-bounded by the M-mode mint
# (CAPCREATE always mints on node 2, which the first section ends, so a second CAPCREATE would be dead on arrival)
lla a2, buf
li t0, 2048
add a2, a2, t0
SPLIT(s11, s2, a2)
lla a2, tohost
lla a3, tohost + 64
CAPBOUND(s11, a2, a3)
lla a2, buf
li t0, 1024
add a2, a2, t0
SPLIT(s7, s2, a2)
lla a2, buf2
lla a3, buf2_end
CAPBOUND(s7, a2, a3)

li s9, 0
la s10, r1
TIGHTEN(s3, s2, x6)            # L2 = duplicate of L (R-21)
r1:
CAPPRINT(s9)                   # #1: 0
li s9, 0
la s10, r2
LCC(t1, s2, x1)                # type - 1 of L: 0 if L is still LINEAR
r2:
CAPPRINT(t1)                   # #2: 0 (the duplicate exists)

DELIN(s3)
li t0, 256
la s10, r3
CDERIVE(s4, s3, t0)            # C on a new node right after N, depth d + 1
r3:
li t3, 0x77
sd t3, 0(s4)

li s9, 0
la s10, r4
MREV(s5, s2)                   # H1
r4:
CAPPRINT(s9)                   # #3: 0
li s9, 0
la s10, r5
MREV(s6, s2)                   # H2
r5:
CAPPRINT(s9)                   # #4: 0

li s9, 0
la s10, r6
REVOKE(s6)                     # ends N's run under H2 (bound d + 1)
r6:
CAPPRINT(s9)                   # #5: 0

li s9, 0
la s10, r7
ld t4, 0(s4)                   # through C
r7:
CAPPRINT(s9)                   # #6: 25 with deepening; 0 = the orphan

li s9, 0
la s10, r8
ld t4, 0(s3)                   # through L2 (N ended)
r8:
CAPPRINT(s9)                   # #7: 25

# the split variant on a fresh region: s7 LINEAR over buf2 (its own node); duplicate, delin, derive, split, one mrev
la s10, r9
TIGHTEN(s8, s7, x6)            # duplicate
r9:
la s10, r9b
DELIN(s8)
r9b:
li t0, 256
la s10, r10
CDERIVE(a4, s8, t0)            # C2 right after N2 at d + 1
r10:
li t3, 0x88
sd t3, 0(a4)
lla a2, buf2
li t0, 2048
add a2, a2, t0
la s10, r11
SPLIT(a5, s7, a2)              # S(d) after N2: N2 -> S -> C2 (sibling after the node)
r11:
la s10, r12
MREV(a6, s7)                   # H3(d) -> N2(d+1) -> S(d) -> C2(d+1)
r12:
li s9, 0
la s10, r13
REVOKE(a6)
r13:
li s9, 0
la s10, r14
ld t4, 0(a4)                   # through C2
r14:
CAPPRINT(s9)                   # #8: 25 with QEMU's sibling placement; 0 with the sibling after the node

#ifdef BOARD_REC_NODE
CAP_PASS(s11)                  # the board wrapper's report; its record capabilities take s11's node (the tohost sibling,
                               # placed before the base's node, outside every walk above), since the base's node is dead
#else
fence
li a0, 1
sw a0, 0(s11)
1: j 1b
#endif

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
  .zero 4096
buf2_end:
  TEST_DATA
RVTEST_DATA_END
