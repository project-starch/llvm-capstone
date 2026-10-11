# loopalign-ctrsanity.S -- E2b of ladder-revival-2026-09-22 (board phases 6-10).
#
# On caplifive_m1_054cea69b the ctrsanity control loop, five 4-byte instructions
# srai/xor/addi/add/bne, costs 7 cycles per iteration when it starts at VA ...1ac and 6
# when it starts at ...1a8, ...1b0, ...1b4 or ...1b8. This test reproduces those five
# start offsets inside a 64-byte sweep window (40, 44, 48, 52, 56; the I-cache line is 16 bytes) and times each loop with
# mcycle, first in plain M-mode, then after capenter, because the board loop ran in
# capability mode. Each arm: 4000 iterations. The RVFI trace records each csrr value.
#
# Reading: per arm, (end mcycle - start mcycle) / 4000 gives cycles per iteration.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.macro LOOPARM off
  csrr t0, mcycle
  li a5, 0
  li a4, 0
  li a3, 4000
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
  LOOPARM 40
  LOOPARM 44
  LOOPARM 48
  LOOPARM 52
  LOOPARM 56

  lla a0, _start
  lla a1, _code_end
  CAPENTER(x0, x0)
  nop
  nop
  nop
  nop

  LOOPARM 40
  LOOPARM 44
  LOOPARM 48
  LOOPARM 52
  LOOPARM 56

  RVTEST_PASS

_code_end:

  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
