# See LICENSE for license details.

#*****************************************************************************
# sup-hostile.S  (supervised CALL on silicon, verification step 7: a hostile domain, and a foreign RETURN)
#-----------------------------------------------------------------------------
#
# Arm 1 (hostile): the supervised domain overwrites every GPR it can reach (ra, sp, gp, tp, t*, s*, a*), stores
# its SEALEDRET into a shared NONLIN region, clobbers ctvec through the legal CCSRRW, and never yields. The
# quantum escapes it (kind 1). The monitor must come back with every register and mtvec intact.
# Arm 2 (foreign RETURN): a second supervised domain LDCs the first domain's SEALEDRET from the shared region
# and RETURNs through it. That RETURN names another seal (base) and must not be honoured: it is an escape with
# kind 2, cause 26 (UNEXPECTED_CAPABILITY_TYPE). The monitor continues after ITS CALL with its registers intact,
# not at the first domain's return point.
#
# Pre-registered readings (CAPPRINT values in the retirement trace):
#   arm 1: status 0, csupstatus 3, csupcause 0x8000000000000010, then the monitor's ra/sp/gp/tp/x30/x8/x22/x31
#          signatures 0x11, 0x22, 1, 0x44, 0xA5, 0x88, 0x1212, 0x1F1F, mtvec - trap_handler = 0, mcause 0
#   arm 2: status 0, csupstatus 5, csupcause 26, csupepc - fault2 = 0, x30 0xA5, x31 0x1F1F, mcause 0
#   the second domain is never resumed; a trap anywhere ends the test with its mcause as the last reading.
# Mutant (positive control): no-foreign-check -> arm 2 "returns" into the first domain's caller context: the
# readings after the arm-2 CALL never appear in this order (the monitor's registers are the first domain's
# seal image), or the run traps.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define CSUPEPC     0xFC2

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

# s1: a NONLIN region both domains inherit (the register leak at first entry) and the first domain stores into
CAPCREATE(s1)
li a0, CAP_TYPE_NONLIN
lla a3, shared
lla a4, shared_end
li a5, CAP_PERM_RWX
CAPTYPE(s1, a0)
CAPBOUND(s1, a3, a4)
CAPPERM(s1, a5)

li t0, 256
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, 0

# ---------------------------------------------------------------- arm 1: the hostile domain
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
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)
# the monitor's signatures (gp stays 1: the harness's pass/fail word)
li ra, 0x11
li sp, 0x22
li tp, 0x44
li x30, 0xA5
li x8, 0x88
li x22, 0x1212
li x31, 0x1F1F
csrw mcause, x0
CSSUPERVISE(t3, a3, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 3 (preempted)
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 0x8000000000000010
addi s2, ra, 0
CAPPRINT(s2)                   # 0x11
addi s2, sp, 0
CAPPRINT(s2)                   # 0x22
addi s2, gp, 0
CAPPRINT(s2)                   # 1
addi s2, tp, 0
CAPPRINT(s2)                   # 0x44
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5 (x30: the readout clobbers t0 = x5)
addi s2, x8, 0
CAPPRINT(s2)                   # 0x88
addi s2, x22, 0
CAPPRINT(s2)                   # 0x1212 (x22: x18 is the readout register)
addi s2, x31, 0
CAPPRINT(s2)                   # 0x1F1F
csrr t0, mtvec
la t1, trap_handler
sub t0, t0, t1
CAPPRINT(t0)                   # 0: mtvec intact
csrr t0, mcause
CAPPRINT(t0)                   # 0

# ---------------------------------------------------------------- arm 2: the foreign RETURN
li gp, 1
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
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a4, a1)
li x30, 0xA5
li x31, 0x1F1F
csrw mcause, x0
li t0, 0x100000
csrw CSUPQUANTUM, t0
CSSUPERVISE(t3, a4, s5)
addi s2, t3, 0
CAPPRINT(s2)                   # status 0
CALL(a4, a4)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5 (fault)
csrr t0, CSUPCAUSE
CAPPRINT(t0)                   # 26
csrr t0, CSUPEPC
la t1, fault2
sub t0, t0, t1
CAPPRINT(t0)                   # 0
addi s2, x30, 0
CAPPRINT(s2)                   # 0xA5 (x30: the readout clobbers t0 = x5)
addi s2, x31, 0
CAPPRINT(s2)                   # 0x1F1F
csrr t0, mcause
CAPPRINT(t0)                   # 0
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause is the last reading
  CAP_PASS(s11)

.align 4
stub1:
  STC(s1, x1, 0)               # park the SEALEDRET where the second domain can find it
  CCSRRW(t0, CCSR_CTVEC, x0)   # legal: the domain's own ctvec (exchanged, slot 1)
  li ra, 0xBAD1
  li sp, 0xBAD2
  li gp, 0xBAD3
  li tp, 0xBAD4
  li t0, 0xBAD5
  li t1, 0xBAD6
  li t2, 0xBAD7
  li s0, 0xBAD8
  li s1, 0xBAD9
  li a0, 0xBADA
  li a1, 0xBADB
  li a2, 0xBADC
  li a3, 0xBADD
  li a4, 0xBADE
  li a5, 0xBADF
  li a6, 0xBAD0
  li a7, 0xBAD1
  li s2, 0xBAD2
  li s3, 0xBAD3
  li s4, 0xBAD4
  li s5, 0xBAD5
  li s6, 0xBAD6
  li s7, 0xBAD7
  li s8, 0xBAD8
  li s9, 0xBAD9
  li s10, 0xBADA
  li s11, 0xBADB
  li t3, 0xBADC
  li t4, 0xBADD
  li t5, 0xBADE
  li t6, 0xBADF
  1: j 1b                      # never yields: the quantum ends it
stub1_end:

.align 4
stub2:
  LDC(x1, s1, 0)               # the first domain's SEALEDRET
  la a1, stub2
fault2:
  RETURN(ra, a1, x0)           # through a foreign seal: an escape, never honoured
  li x20, 0xDEAD
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
shared:
  .zero 64
shared_end:
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
