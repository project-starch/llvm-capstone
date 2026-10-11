# cap-atomics.S -- do the atomics musl and memcached use work through a CAPABILITY address in capability mode?
#
# memcached.dom links 171 atomics: lr.w x80 and sc.w x80 (musl's a_cas loops), amoadd.w x4, amoadd.d x5 and
# amoswap.w x2. No image proven on silicon contains any (speedtest1, k800: zero AMO-opcode words), and no RTL
# directed test runs one through a capability in capability mode (r43-evict-live and s06sec-amo-no-resurrect use
# integer addresses). The tag half (an AMO onto a capability granule leaves its tag set, invariant I4) is a
# known RTL residual with its own repro, s06sec-amo-no-resurrect.S, and is NOT tested here.
#
# Every reading is architectural, with nothing timing-dependent. The pre-registered vector (CAPPRINT order) is in
# the folder's PREREG.md.
# Registers: data capabilities s1 (x9), s2 (x18), s3 (x19); scratch t0..t6. The board recorder uses x8, x16 and
# x17 (build.sh), and the report uses x23..x31 after the last reading.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc
.section .text.init
.globl _start
_start:
  li gp, 1
  csrwi mie, 0
  la t0, trap_handler
  csrw mtvec, t0

  lla t0, _start
  lla t1, _end_of_text
  CAPENTER(t0, t1)

  # s1: NONLIN RW over buf (64 B), cursor at +0; s2 = s1 + 32 (a word); s3 = s1 + 48 (another granule)
  CAPCREATE(s1)
  li t0, CAP_TYPE_NONLIN
  lla t1, buf_start
  lla t2, buf_end
  li t3, CAP_PERM_RW
  CAPTYPE(s1, t0)
  CAPBOUND(s1, t1, t2)
  CAPPERM(s1, t3)
  CINCOFFSETIMM(s2, s1, 32)
  CINCOFFSETIMM(s3, s1, 48)

  # 1: plain store/load through the capability (control)
  li t0, 0x1111
  sd t0, 0(s1)
  ld t1, 0(s1)
  CAPPRINT(t1)                       # 1  0x1111
  # 2-5: amoadd.d / amoswap.d
  li t2, 5
  amoadd.d t1, t2, (s1)
  CAPPRINT(t1)                       # 2  0x1111 (old)
  ld t1, 0(s1)
  CAPPRINT(t1)                       # 3  0x1116
  li t2, 0xABCD
  amoswap.d t1, t2, (s1)
  CAPPRINT(t1)                       # 4  0x1116 (old)
  ld t1, 0(s1)
  CAPPRINT(t1)                       # 5  0xABCD
  # 6-9: amoadd.w (32-bit wrap, sign-extended result) / amoswap.w
  li t0, 0x7fffffff
  sw t0, 0(s2)
  li t2, 1
  amoadd.w t1, t2, (s2)
  CAPPRINT(t1)                       # 6  0x7fffffff (old)
  lw t1, 0(s2)
  CAPPRINT(t1)                       # 7  0xffffffff80000000
  li t2, 0x12345678
  amoswap.w t1, t2, (s2)
  CAPPRINT(t1)                       # 8  0xffffffff80000000 (old)
  lw t1, 0(s2)
  CAPPRINT(t1)                       # 9  0x12345678
  # 10-14: lr.w/sc.w success, then sc.w with no reservation (the successful sc consumed it)
  lr.w t1, (s2)
  CAPPRINT(t1)                       # 10 0x12345678
  li t2, 0x55
  sc.w t3, t2, (s2)
  CAPPRINT(t3)                       # 11 0 (success)
  lw t1, 0(s2)
  CAPPRINT(t1)                       # 12 0x55
  li t2, 0x66
  sc.w t3, t2, (s2)
  CAPPRINT(t3)                       # 13 non-zero (no reservation)
  lw t1, 0(s2)
  CAPPRINT(t1)                       # 14 0x55 (unchanged)
  # 15-16: lr.w on one granule, sc.w on another: must fail and write nothing
  li t0, 0xB0B0
  sw t0, 0(s3)
  lr.w t1, (s2)
  li t2, 0x77
  sc.w t3, t2, (s3)
  CAPPRINT(t3)                       # 15 non-zero (address outside the reservation)
  lw t1, 0(s3)
  CAPPRINT(t1)                       # 16 0xB0B0 (unchanged)
  # 17-19: lr.d/sc.d success
  lr.d t1, (s1)
  CAPPRINT(t1)                       # 17 0xABCD
  li t2, 0x1234
  sc.d t3, t2, (s1)
  CAPPRINT(t3)                       # 18 0 (success)
  ld t1, 0(s1)
  CAPPRINT(t1)                       # 19 0x1234
  # 20-21: musl-style a_cas loop on the word: expect 0x55, store 0x99; count the attempts
  li t4, 0
  li t5, 0x55
  li t6, 0x99
1:
  addi t4, t4, 1
  lr.w t1, (s2)
  bne t1, t5, 2f
  sc.w t3, t6, (s2)
  bnez t3, 1b
2:
  CAPPRINT(t4)                       # 20 1 (one attempt)
  lw t1, 0(s2)
  CAPPRINT(t1)                       # 21 0x99
  li t0, 0xA70D
  CAPPRINT(t0)                       # 22 0xA70D (end marker)
  CAP_PASS(s1)

  .align 4
trap_handler:
  # Any trap is a failure: record mcause, mepc, mtval and the sentinel 0xDEAD, then report.
  csrr t0, mcause
  CAPPRINT(t0)
  csrr t0, mepc
  CAPPRINT(t0)
  csrr t0, mtval
  CAPPRINT(t0)
  li t0, 0xDEAD
  CAPPRINT(t0)
  CAP_PASS(s1)

.align 4
.globl _end_of_text
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
buf_start:
  .zero 64
buf_end:
  TEST_DATA
RVTEST_DATA_END
