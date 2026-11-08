# cap-type-word.S -- which type NUMBER does the RTL write into a stored capability's second doubleword?
# QEMU stores its enum (LIN 0, NONLIN 1; cap.h, cap_compress.c ty field [29:27]); the RTL enum is NOT_CAP 0,
# LINEAR 1, NONLIN 2 (ariane_pkg.sv). Arms: N = a NONLIN capability, L = a LINEAR one, each STC'd, then the hi
# word read back with a plain ld from an integer address. Prints per arm: hi word, (hi >> 27) & 7, LCC type.
# Prediction (RTL numbering stored raw): N 2, L 1; LCC (spec numbering) N 1, L 0. A hi word of 0 = VOID
# (the plain load does not return metadata).
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"
.option norvc
.section .text.init
.globl _start
_start:
  li gp, 1
#define MKCAP(CREG, TY)      \
  CAPCREATE(CREG);           \
  li  a0, TY;                \
  lla a3, buf_start;         \
  lla a4, buf_end;           \
  li  a5, CAP_PERM_RWX;      \
  CAPTYPE(CREG, a0);         \
  CAPBOUND(CREG, a3, a4);    \
  CAPPERM(CREG, a5);
  MKCAP(a1, CAP_TYPE_NONLIN)           /* address capability for the stores */
  MKCAP(s3, CAP_TYPE_NONLIN)
  li   t0, 1
  LCC(s4, s3, x1)                      /* type, LCC numbering */
  STC(a1, s3, 0x40)
  lla  t1, buf_start
  ld   s5, 0x48(t1)
  srli s6, s5, 27
  andi s6, s6, 7
  CAPPRINT(s5)
  CAPPRINT(s6)
  CAPPRINT(s4)
  MKCAP(s7, CAP_TYPE_LIN)
  LCC(s8, s7, x1)
  STC(a1, s7, 0x80)
  ld   s9, 0x88(t1)
  srli s10, s9, 27
  andi s10, s10, 7
  CAPPRINT(s9)
  CAPPRINT(s10)
  CAPPRINT(s8)
  RVTEST_PASS
fail:
  RVTEST_FAIL
  .data
RVTEST_DATA_BEGIN
.align 6
buf_start:
  .zero 256
buf_end:
  TEST_DATA
RVTEST_DATA_END
