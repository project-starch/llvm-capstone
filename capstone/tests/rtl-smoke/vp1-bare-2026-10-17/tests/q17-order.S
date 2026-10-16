# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp.

#*****************************************************************************
# q17-order.S  (Q-17, 2026-10-16: cderive's check order, the liveness query before the size window)
#-----------------------------------------------------------------------------
#
# A NONLIN parent over `data`; a child derived from it; the child crevoked. Then two cderives from the DEAD child, each
# trapping (the handler records mcause and steps over the instruction):
#   #1 cderive(dead child, 4096): the window is outside the child AND the child is dead. QEMU reads 25 (helper_csderive
#      queries liveness before the window); the first Sublet RTL read 29 (window first). Pre-registered: 25 on the
#      tree with the Q-17 reorder, 29 on capstone-sublet b7e3287e4 / caplifive_sublet_d93a8d66b (the positive control).
#   #2 cderive(dead child, 16): inside the window, dead: 25 on both trees.
#   #3 the trap count: 2.
# Then CAP_PASS. A trap anywhere else changes #3.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s9, 0
  li   s10, 0

  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)

  CAPCREATE(s2)
  li   t0, CAP_TYPE_NONLIN
  lla  t1, data
  lla  t2, data_end
  CAPTYPE(s2, t0)
  CAPBOUND(s2, t1, t2)
  li   t0, 14
  CAPPERM(s2, t0)
  sd   x0, 0(s2)

  li   t0, 64
  CDERIVE(s3, s2, t0)           # the child over [data, data + 64)
  CREVOKE(s2, s3)               # dead

  li   s9, 0
  li   t0, 4096
  CDERIVE(s4, s3, t0)           # #1: dead AND outside the window
  CAPPRINT(s9)
  li   s9, 0
  li   t0, 16
  CDERIVE(s4, s3, t0)           # #2: dead, inside the window
  CAPPRINT(s9)
  CAPPRINT(s10)                 # #3: 2
  CAP_PASS(s11)

.align 4
trap_handler:
  csrr s9, mcause
  addi s10, s10, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 6
data:
  .zero 4096
data_end:
RVTEST_DATA_END
