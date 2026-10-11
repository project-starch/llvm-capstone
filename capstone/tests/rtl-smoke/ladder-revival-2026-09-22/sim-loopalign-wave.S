# loopalign-wave.S -- minimal waveform companion to loopalign-ctrsanity.S (E2b).
# The same five-instruction loop at offsets 44 (slow) and 52 (fast) of a 64-byte line,
# 16 iterations each, plain M-mode only, so the run passes and the waveform stays small.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.macro LOOPARM off
  csrr t0, mcycle
  li a5, 0
  li a4, 0
  li a3, 16
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
  csrr t1, mcycle
  sub t2, t1, t0
  CHK_START
.endm

.section .text.init
.global _start

_start:
  LOOPARM 44
  LOOPARM 52
  RVTEST_PASS

  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
