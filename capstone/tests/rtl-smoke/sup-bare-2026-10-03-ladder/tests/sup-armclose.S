# See LICENSE for license details.

#*****************************************************************************
# sup-armclose.S  (supervised CALL on silicon: an armed cssupervise FAILS CLOSED, and the arm PERSISTS)
#-----------------------------------------------------------------------------
#
# While a cssupervise is armed for seal A, a CALL of any other seal B must not run (it would run unsupervised,
# with no quantum and full M-mode CSR access, and nothing would say so). As QEMU's capstone_supervisor_call
# does, the mismatched CALL traps as an illegal instruction and the arm persists -- across the trap, across the
# handler's mret, and across any exception a CALL head takes (a type fault here; an interrupt or a debug request
# on the board). Only an acknowledged armed CALL or the forget form (cssupervise x0) clears it.
#
# Flow: arm A; CALL B traps (cause 2 at the CALL); the handler mrets WITHOUT forgetting, so the re-executed CALL
# traps again (the arm survived); the second time the handler forgets and mrets to the UNCHANGED mepc: the same
# CALL now runs B ordinarily (B stores 0xBB, RETURNs to CALL + 4). Then: arm A; a CALL of a LINEAR (unsealed)
# capability faults at the CALL; the handler steps over it without forgetting; CALL A must then be SUPERVISED:
# its ecall escapes (csupstatus 5, csupcause 11) instead of reaching the trap handler.
#
# Pre-registered readings (CAPPRINT values in the retirement trace), 2026-10-01 evening:
#   #1 0 (armed for A)
#   #2 2, #3 0 (the CALL of B trapped, mepc = callB)         -- the handler mrets, arm kept
#   #4 2, #5 0 (it trapped AGAIN: the arm survived the trap and the mret)   #6 0 (the forget form's status)
#   #7 0xBB (B ran ordinarily from the re-executed CALL, after the forget)
#   #8 0 (armed for A)   #9 the planted CALL's cause (27, UNEXPECTED_CAP_TYPE, as the dyn unit raises it: any value
#   but 2 and 11 takes the step-over path)   #10 0 (mepc = callL)
#   #11 5, #12 11 (A's ecall ESCAPED: the arm survived the planted fault)   #13 0xBB (B's marker intact)   #14 0 (mcause)
# Mutants (positive controls, sup-variant.py):
#   no-armclose      -> fail open again: #2..#6 never appear, B runs unsupervised from the first CALL, #7 follows #1
#   no-armclose-trap -> the switch is blocked but no trap raised: the CALL wedges at the head, nothing after #1
#   disarm-on-fault  -> the planted fault drops the arm: after #10 the handler prints 11 (A ran UNSUPERVISED, its ecall
#                       trapped) and the test ends there; #11..#14 never appear

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li s8, 0                       # the handler's count of cause-2 traps

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

CAPCREATE(s5)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s5, a0)
CAPBOUND(s5, a3, a4)
CAPPERM(s5, a5)

CAPCREATE(s1)
li a0, CAP_TYPE_NONLIN
lla a3, marker
lla a4, marker_end
li a5, CAP_PERM_RWX
CAPTYPE(s1, a0)
CAPBOUND(s1, a3, a4)
CAPPERM(s1, a5)
sd x0, 0(s1)

# seal A (stubA: ecall) in s6; seal B (stubB: store 0xBB, RETURN) in s7
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stubA
lla a4, stubA_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, sealA
lla a4, sealA_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(s6, a1)
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stubB
lla a4, stubB_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, sealB
lla a4, sealB_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(s7, a1)

# a6: a LINEAR capability that is not a seal (the planted CALL fault)
CAPCREATE(a6)
li a0, CAP_TYPE_LIN
lla a3, savearea2
lla a4, savearea2_end
li a5, CAP_PERM_RWX
CAPTYPE(a6, a0)
CAPBOUND(a6, a3, a4)
CAPPERM(a6, a5)

li t0, 0x100000
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, 0

CSSUPERVISE(t3, s6, s5)        # arm for A
addi s2, t3, 0
CAPPRINT(s2)                   # #1 0
callB:
CALL(s7, s7)                   # a CALL of B while armed for A: TRAPS (cause 2), twice, then runs ordinarily after the forget
ld t0, 0(s1)
CAPPRINT(t0)                   # #7 0xBB
CSSUPERVISE(t3, s6, s5)        # arm for A again
addi s2, t3, 0
CAPPRINT(s2)                   # #8 0
callL:
CALL(a6, a6)                   # a CALL of a LINEAR capability: faults at the CALL; the arm must survive it
CALL(s6, s6)                   # A: ecall -> escape (supervised, because the arm survived)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # #11 5
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # #12 11
ld t0, 0(s1)
CAPPRINT(t0)                   # #13 0xBB still
csrr t0, mcause
CAPPRINT(t0)                   # #14 0
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # #2 / #4 2 (the mismatched CALL), #9 the planted fault's cause; 11 only if A ran unsupervised
  li t1, 11
  beq s4, t1, unsupervised     # an ecall reaching the handler: A ran without supervision (the disarm-on-fault mutant)
  li t1, 2
  bne s4, t1, planted
  addi s8, s8, 1
  csrr s4, mepc
  la t1, callB
  sub s4, s4, t1
  CAPPRINT(s4)                 # #3 / #5 0: at the CALL itself
  li t1, 2
  blt s8, t1, back             # the first time: mret with the arm KEPT -- the re-executed CALL must trap again
  CSSUPERVISE(t3, x0, x0)      # the second time: forget the arm
  addi s2, t3, 0
  CAPPRINT(s2)                 # #6 0
back:
  csrw mcause, x0
  mret                         # mepc unchanged: the CALL at callB re-executes
planted:
  csrr s4, mepc
  la t1, callL
  sub s4, s4, t1
  CAPPRINT(s4)                 # #10 0: the planted fault is at callL
  csrr s4, mepc
  addi s4, s4, 4               # step over it; the arm is NOT forgotten
  csrw mepc, s4
  csrw mcause, x0
  mret
unsupervised:
  CAP_PASS(s11)                # the test ends here: 11 is its last reading

.align 4
stubA:
  ecall
  1: j 1b
stubA_end:

.align 4
stubB:
  li t0, 0xBB
  sd t0, 0(s1)
  la a1, stubB
  RETURN(ra, a1, x0)
stubB_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
savearea2:
  .zero 1024
savearea2_end:
.align 6
marker:
  .zero 64
marker_end:
.align 6
sealA:
  .zero 2048
sealA_end:
.align 6
sealB:
  .zero 2048
sealB_end:
  TEST_DATA
RVTEST_DATA_END
