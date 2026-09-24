# loopalign-warm.S -- discriminator for the E2b hypothesis. Arm A: the slow loop (offset 44)
# as-is. Arm B: identical, but the 64-byte line FOLLOWING the loop is executed once before
# the loop (a jump into it and straight back), so it is resident in the I-cache when the
# loop's speculative next-line fetch reaches it. Hypothesis: B runs at 6 cycles/iteration.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.macro LOOPARM off, warm
  li a5, 0
  li a4, 0
  li a3, 64
.if \warm
  la ra, 4f
  j 5f
4:
.endif
  csrr t0, mcycle
  j 3f
  .balign 64
  .rept (\off / 4)
  nop
  .endr
3:
  srai a6, a4, 3
  xor a6, a6, a5
  addi a5, a5, 1
  add a4, a6, a4
  bne a3, a5, 3b
5:
.if \warm
  bnez a5, 6f
  jr ra
6:
.endif
  csrr t1, mcycle
  sub t2, t1, t0
  CHK_START
.endm

.section .text.init
.global _start

_start:
  LOOPARM 44, 0
  LOOPARM 44, 1
  LOOPARM 52, 0
  RVTEST_PASS

  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
