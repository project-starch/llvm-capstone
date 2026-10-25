# See LICENSE for license details.

#*****************************************************************************
# sup-sublet.S  (CDERIVE / CREVOKE inside a supervised domain, 2026-10-14)
#-----------------------------------------------------------------------------
#
# The Sublet instructions are allowed to a supervised domain (decoder.sv: the deny list covers only the mints), so a
# domain must be able to derive and end its own children, and a refused CDERIVE / CREVOKE must leave the domain the
# way any capability fault does: a kind-2 escape to the monitor with the cause in CSUPCAUSE, never a trap, never a
# domain that runs on. Five arms on the sup-escape.S pattern, each with its own seal and stub (a faulted domain is
# never CALLed again). The domain inherits its capabilities in GPRs at first entry: x26 = the NONLIN RW+MANAGE
# parent over `data`, s1 = a LINEAR RW scratch the monitor reads back after a RETURN. Nothing is minted inside.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   arm 0 (returns): status 0 (armed); inside the domain: child perm 14 (0xe), the value stored and loaded through
#     the child 0x77, 0 after crevoke(parent, child) (the kill's own status: the instruction retired); after RETURN:
#     csupstatus 1 (kind 0, valid), the three words the domain stored through s1: 0x77, 0xe, 0x5A5A
#   arm 1: status 0; csupstatus 5 (fault); csupcause 25 (a load through the child right after crevoke: R-45's flush
#     refetches it and the node is dead); csupepc - fault1 = 0; lcc(a3) = 4 (SEALED)
#   arm 2: status 0; 5; csupcause 27 (crevoke through a tightened parent without MANAGE); 0; 4
#   arm 3: status 0; 5; csupcause 29 (crevoke of a grandchild: not a direct child); 0; 4
#   arm 4: status 0; 5; csupcause 26 (cderive from the LINEAR s1); 0; 4
#   mcause 0 (no trap was taken anywhere)
# A trap anywhere ends the test with the trap handler's mcause as the last reading.
# Instruction names are lower case in comments because this file goes through cpp.
# On the board (BOARD_TEST_CH defined by the bare wrapper) the three in-domain CAPPRINTs are compiled out, so the
# board vector is the 29 readings above minus the domain's 0xe, 0x77 and 0: 26 values, the monitor's reads of
# 0x77 / 0xe / 0x5a5a through s1 carrying the same information.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define CSUPEPC     0xFC2
#define CSUPTVAL    0xFC3

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

# s5: the monitor's private save area, LINEAR RWX, 1 KiB, 64-byte aligned
CAPCREATE(s5)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s5, a0)
CAPBOUND(s5, a3, a4)
CAPPERM(s5, a5)

# x26: the parent the domains derive from, NONLIN RW + MANAGE over `data` (inherited at first entry)
CAPCREATE(x26)
li a0, CAP_TYPE_NONLIN
lla a3, data
lla a4, data_end
li a5, 14
CAPTYPE(x26, a0)
CAPBOUND(x26, a3, a4)
CAPPERM(x26, a5)
sd x0, 0(x26)
sd x0, 8(x26)

# s1: a LINEAR RW scratch the domain stores its readings through, and the LINEAR parent of arm 4
CAPCREATE(s1)
li a0, CAP_TYPE_LIN
lla a3, scratch
lla a4, scratch_end
li a5, CAP_PERM_RW
CAPTYPE(s1, a0)
CAPBOUND(s1, a3, a4)
CAPPERM(s1, a5)
sd x0, 0(s1)
sd x0, 8(s1)
sd x0, 16(s1)

li t0, 0x100000
csrw CSUPQUANTUM, t0           # a quantum far longer than any arm
csrwi CSUPCTL, 0               # first entries, never a resume

#define ARM_SETUP(STUB, STUB_END, SEALX, SEALX_END) \
  CAPCREATE(a2); li a0, CAP_TYPE_LIN; lla a3, STUB; lla a4, STUB_END; li a5, CAP_PERM_RWX; \
  CAPTYPE(a2, a0); CAPBOUND(a2, a3, a4); CAPPERM(a2, a5); \
  CAPCREATE(a1); li a0, CAP_TYPE_LIN; lla a3, SEALX; lla a4, SEALX_END; li a5, CAP_PERM_RWX; \
  CAPTYPE(a1, a0); CAPBOUND(a1, a3, a4); CAPPERM(a1, a5); \
  STC(a1, a2, 0); li a0, 0x0000000A00040000; sd a0, 48(a1); sd x0, 56(a1); sd x0, 64(a1); sd x0, 72(a1); sd x0, 80(a1); \
  SEAL(a3, a1); csrw mcause, x0; CSSUPERVISE(t3, a3, s5); addi s2, t3, 0; CAPPRINT(s2); CALL(a3, a3)

#define ARM_FAULT_READOUT(FAULTK) \
  csrr t0, CSUPSTATUS; CAPPRINT(t0); csrr t0, CSUPCAUSE; CAPPRINT(t0); \
  csrr t0, CSUPEPC; la t1, FAULTK; sub t0, t0, t1; CAPPRINT(t0); LCC(s3, a3, x1); CAPPRINT(s3)

# ---------------------------------------------------------------- arm 0: derive, use, crevoke, RETURN
ARM_SETUP(stub0, stub0_end, seal0, seal0_end)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 1: kind 0 = the RETURN, valid
ld t0, 0(s1)
CAPPRINT(t0)                   # 0x77
ld t0, 8(s1)
CAPPRINT(t0)                   # 0xe
ld t0, 16(s1)
CAPPRINT(t0)                   # 0x5A5A

# ---------------------------------------------------------------- arm 1: a load through the child after crevoke -> 25
ARM_SETUP(stub1, stub1_end, seal1, seal1_end)
ARM_FAULT_READOUT(fault1)      # 5, 25, 0, 4

# ---------------------------------------------------------------- arm 2: crevoke without MANAGE -> 27
ARM_SETUP(stub2, stub2_end, seal2, seal2_end)
ARM_FAULT_READOUT(fault2)      # 5, 27, 0, 4

# ---------------------------------------------------------------- arm 3: crevoke of a grandchild -> 29
ARM_SETUP(stub3, stub3_end, seal3, seal3_end)
ARM_FAULT_READOUT(fault3)      # 5, 29, 0, 4

# ---------------------------------------------------------------- arm 4: cderive from a LINEAR -> 26
ARM_SETUP(stub4, stub4_end, seal4, seal4_end)
ARM_FAULT_READOUT(fault4)      # 5, 26, 0, 4

csrr t0, mcause
CAPPRINT(t0)                   # 0
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause is the last reading
  CAP_PASS(s11)

# ---- the domains. No mint, no CSR, no CALL: sup-static-audit.py must stay silent on every stub.
.align 4
stub0:
  MOVC(s3, x1)                 # park the SEALEDRET
  li t0, 256
  CDERIVE(s4, x26, t0)
  LCC(t1, s4, x5)
  # the three in-domain CAPPRINTs are simulation only: the board wrapper's CAPPRINT mints a record capability, which a
  # supervised domain may not do; the monitor re-reads these values through s1 below
#ifndef BOARD_TEST_CH
  CAPPRINT(t1)                 # 0xe
#endif
  li t2, 0x77
  sd t2, 0(s4)
  ld t2, 0(s4)
#ifndef BOARD_TEST_CH
  CAPPRINT(t2)                 # 0x77
#endif
  sd t2, 0(s1)
  sd t1, 8(s1)
  CREVOKE(x26, s4)
  li t2, 0
#ifndef BOARD_TEST_CH
  CAPPRINT(t2)                 # 0: the kill retired
#endif
  li t2, 0x5A5A
  sd t2, 16(s1)
  MOVC(x1, s3)
  la a1, stub0
  RETURN(ra, a1, x0)
stub0_end:

.align 4
stub1:
  li t0, 256
  CDERIVE(s4, x26, t0)
  CREVOKE(x26, s4)
fault1:
  ld t2, 0(s4)
  li x20, 0xDEAD               # never reached: the fault escapes to the monitor
  1: j 1b
stub1_end:

.align 4
stub2:
  li t0, 256
  CDERIVE(s4, x26, t0)
  TIGHTEN(a4, x26, x6)         # RW without MANAGE
fault2:
  CREVOKE(a4, s4)
  li x20, 0xDEAD
  1: j 1b
stub2_end:

.align 4
stub3:
  li t0, 256
  CDERIVE(s4, x26, t0)
  li t0, 64
  CDERIVE(s6, s4, t0)          # the grandchild
fault3:
  CREVOKE(x26, s6)
  li x20, 0xDEAD
  1: j 1b
stub3_end:

.align 4
stub4:
  li t0, 64
fault4:
  CDERIVE(s4, s1, t0)          # s1 is LINEAR
  li x20, 0xDEAD
  1: j 1b
stub4_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
scratch:
  .zero 64
scratch_end:
.align 6
data:
  .zero 4096
data_end:
.align 6
seal0:
  .zero 2048
seal0_end:
.align 6
seal1:
  .zero 2048
seal1_end:
.align 6
seal2:
  .zero 2048
seal2_end:
.align 6
seal3:
  .zero 2048
seal3_end:
.align 6
seal4:
  .zero 2048
seal4_end:
  TEST_DATA
RVTEST_DATA_END
