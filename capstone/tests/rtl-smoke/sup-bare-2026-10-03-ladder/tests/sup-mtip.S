# See LICENSE for license details.

#*****************************************************************************
# sup-mtip.S  (supervised CALL on silicon, verification step 5: a real interrupt inside the domain)
#-----------------------------------------------------------------------------
#
# The CLINT timer is armed a little ahead of mtime. Arm 1: the domain's seal image enables the timer interrupt
# (mstatus.MIE = 1, mie.MTIE = 1); the domain spins; MTIP is marked on one of its instructions and must escape
# to the monitor as kind 1 with the M-timer cause 0x8000000000000007 -- never a trap, never a handler run under
# the domain's PC capability. Arm 2: the same with mie = 0 in the image and a 256-cycle quantum: the interrupt
# stays masked and the quantum fires instead (cause 0x8000000000000010). The monitor itself runs with MIE = 0.
#
# Pre-registered readings (CAPPRINT values in the retirement trace):
#   reference: the spin count until the monitor's own timer trap, then 0x8000000000000007
#   arm 1: status 0, csupstatus 3, csupcause 0x8000000000000007, x30 0xA5, mcause 0, mip & MTIP (bit 7) = 0x80,
#          then after a resume with the timer re-armed: csupstatus 3 and the timer cause again (the domain's mie survived)
#   arm 2: status 0, csupstatus 3, csupcause 0x8000000000000010, x30 0xA5, mcause 0
#   a trap anywhere ends the test with its mcause as the last reading.

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

# s6: mtime, s8: mtimecmp (the CLINT of the test harness)
CAPCREATE(s6)
li a0, CAP_TYPE_LIN
li a3, 0x0200BFF8
li a4, 0x0200C000
li a5, CAP_PERM_RWX
CAPTYPE(s6, a0)
CAPBOUND(s6, a3, a4)
CAPPERM(s6, a5)
CAPCREATE(s8)
li a0, CAP_TYPE_LIN
li a3, 0x02004000
li a4, 0x02004008
li a5, CAP_PERM_RWX
CAPTYPE(s8, a0)
CAPBOUND(s8, a3, a4)
CAPPERM(s8, a5)

# reference: the monitor itself takes the timer interrupt as an ordinary trap (mcause 0x8000000000000007), so the
# CLINT, mip, mie and the decoder's marking are known to work in this harness before a domain is asked to show it;
# the handler prints mcause and the number of spin iterations it took (an upper bound on the tick rate), then mret
li t0, 0x80
csrw mie, t0
csrsi mstatus, 8
ld t0, 0(s6)
addi t0, t0, 400
sd t0, 0(s8)
li s7, 0
li s9, 0
li t1, 4000000
ref_spin:
  addi s7, s7, 1
  beq s9, x0, ref_spin_more
  j ref_done
ref_spin_more:
  bne s7, t1, ref_spin
ref_done:
addi s2, s7, 0
CAPPRINT(s2)                   # reference: iterations until the interrupt (4000000 = it never came)
addi s2, s9, 0
CAPPRINT(s2)                   # reference: 0x8000000000000007 (set by the handler) or 0
csrw mie, x0
csrci mstatus, 8
csrw mcause, x0

# ---------------------------------------------------------------- arm 1: MTIP escapes
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub1
lla a4, stub1_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal1
lla a4, seal1_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040008
sd a0, 48(a1)                  # mstatus image: MIE = 1
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
li a0, 0x80
sd a0, 80(a1)                  # mie image: MTIE
SEAL(a3, a1)
li t0, 0x100000
csrw CSUPQUANTUM, t0           # the quantum must not be what ends arm 1
csrwi CSUPCTL, 0
li x30, 0xA5
csrw mcause, x0
ld t0, 0(s6)
addi t0, t0, 400
sd t0, 0(s8)                   # mtimecmp := mtime + 400
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 3 (preempted)
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 0x8000000000000007
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5
csrr t0, mcause
CAPPRINT(t0)                   # 0: the monitor never took the interrupt either (its MIE is 0)
csrr t0, mip
andi t0, t0, 0x80
CAPPRINT(t0)                   # 0x80: still pending
# resume the domain with the timer re-armed: its mie (MTIE) and mstatus (MIE) must come back from its seal
ld t0, 0(s6)
addi t0, t0, 400
sd t0, 0(s8)
csrwi CSUPCTL, 1
CSSUPERVISE(t3, a3, s5)
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 3 again: a second timer escape after the resume
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 0x8000000000000007 again
csrwi CSUPCTL, 0
li t0, -1
sd t0, 0(s8)                   # disarm the timer

# ---------------------------------------------------------------- arm 2: masked in the image, the quantum fires
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub2
lla a4, stub2_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal2
lla a4, seal2_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040008
sd a0, 48(a1)                  # mstatus image: MIE = 1
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)                  # mie image: nothing enabled
SEAL(a3, a1)
li t0, 256
csrw CSUPQUANTUM, t0
csrw mcause, x0
ld t0, 0(s6)
addi t0, t0, 100
sd t0, 0(s8)                   # the timer fires inside the domain again, but is masked there
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 3
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 0x8000000000000010
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5
csrr t0, mcause
CAPPRINT(t0)                   # 0
li t0, -1
sd t0, 0(s8)
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  bltz s4, ref_interrupt       # the reference arm's timer interrupt: record it and return
  CAPPRINT(s4)                 # any other trap ends the test: its cause is the last reading
  CAP_PASS(s11)
ref_interrupt:
  addi s9, s4, 0
  li t0, -1
  sd t0, 0(s8)                 # disarm, so that mret does not re-enter
  mret

.align 4
stub1:
  li x30, 0xBAD5
  1: j 1b
stub1_end:

.align 4
stub2:
  li x30, 0xBAD5
  1: j 1b
stub2_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
seal1:
  .zero 2048
seal1_end:
.align 6
seal2:
  .zero 2048
seal2_end:
  TEST_DATA
RVTEST_DATA_END
