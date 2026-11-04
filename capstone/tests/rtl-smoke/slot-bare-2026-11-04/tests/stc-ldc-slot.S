# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# stc-ldc-slot.S -- what a capability copy through one stack slot costs, against a load of a DIFFERENT slot (2026-11-04)
#-----------------------------------------------------------------------------
#
# The SPEC lane's question: devmk (+movc-keeps-integer-source) removes "stc then ldc of the same slot" pairs, and on
# silicon that removes ~3 % of instructions and 10-12 % of cycles (imagick, bitstream 3 and ObjBounds alike). Is the
# pair's cost the store-to-load hazard, and is it specific to capabilities?
#
# SOURCE (read before the run): the load unit has no store-to-load forwarding. Any load whose page offset [11:3]
# matches a valid entry of the store buffer's speculative or commit queue, or the store in flight, waits in
# WAIT_PAGE_OFFSET until no entry matches (store_buffer.sv page_offset_matches_o; load_unit.sv). The compare reads the
# page offset only, so a store to another page at the same offset also stalls the load, and the compare does not look
# at the access type, so a plain sd/ld pair should pay the same wait.
#
# HYPOTHESIS: the same-slot pair's cost is that wait; a load of a different slot does not wait.
# Arms, -DSL=n, each 512 iterations of <body>; addi; bnez. Slot A = heap+0x2000.
#   SL 1  stc A; ldc A                       (= objb-arcfour ARC 3)   predict: b3 ~20.5/iter, ObjBounds ~11.5/iter
#                                                                     (ARC 3's readings), C.5a within +3/iter of ObjBounds
#   SL 2  stc A; ldc A+0x10 (other doubleword, same line)             predict: <= half of SL 1 on every tree
#   SL 3  stc A; ldc A+0x40 (other cache line)                        predict: = SL 2 within 1/iter
#   SL 4  stc A; ldc A+0x1000 (other page, SAME page offset)          predict: = SL 1 within 1/iter (offset-only compare)
#   SL 5  sd A; ld A    (plain, through the frame capability)         predict: >= SL 6 + 5/iter; equal across trees within 1
#   SL 6  sd A; ld A+0x10                                             predict: <= half of SL 5
#   SL 7  ldc A alone                                                 predict: the LDC's own cost (b3 > ObjBounds)
#   SL 8  stc A alone                                                 predict: the STC's own cost
#   SL 9  the loop alone (addi; bnez)                                 predict: ~2/iter, equal on every tree
# A forwarding path would save at most SL 1 - SL 2 per pair. Each arm prints the mcycle and minstret deltas over the
# loop (minstret is the control: equal on every tree), a value, then the trap count and last cause (0, 0). The value:
# for the ldc arms, a byte loaded THROUGH the loop's last ldc result (0x77; a3 starts at a byte holding 0); for SL 5
# the last ld (1, the last counter stored); for SL 6 the prefilled 0x5a; for SL 8 and SL 9, 0.
# Memory delay is the testbench's (S12_MEM_DELAY=12), not the board's DDR: the cycles are a pattern, not silicon costs.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#ifndef SL
#define SL 1
#endif
#define N 512

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s4, 0
  li   s10, 0

  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)

  CAPCREATE(s2)
  li   t1, CAP_TYPE_NONLIN
  lla  a2, heap
  lla  a3, heap_end
  CAPTYPE(s2, t1)
  CAPBOUND(s2, a2, a3)
  li   t1, CAP_PERM_ALL
  CAPPERM(s2, t1)

  li   t1, 1024
  CINCOFFSET(t0, s2, t1)         # t0: the capability that is copied, at heap+1024 (holds 0x77)
  li   t1, 0x77
  sb   t1, 0(t0)
  li   t1, 1025
  CINCOFFSET(a3, s2, t1)         # a3 starts at heap+1025 (holds 0): 0x77 can come only from a loop ldc
  li   t1, 0x2030
  CINCOFFSET(s0, s2, t1)         # frame: slot A is -0x30(s0) = heap+0x2000
  li   t1, 0x3030
  CINCOFFSET(s1, s2, t1)         # the page alias: -0x30(s1) = heap+0x3000, the same page offset as A

  STC(s0, t0, -0x30)             # prefill A, A+0x10, A+0x40 and the alias with t0
  STC(s0, t0, -0x20)
  STC(s0, t0, 0x10)
  STC(s1, t0, -0x30)
#if SL == 5 || SL == 6
  li   t1, 0x5a
  sd   t1, -0x30(s0)
  sd   t1, -0x20(s0)
#endif
  li   a5, 0
  fence
  csrr s5, mcycle
  csrr s6, minstret
  li   t2, N

3:
#if SL == 1
  STC(s0, t0, -0x30)
  LDC(a3, s0, -0x30)
#elif SL == 2
  STC(s0, t0, -0x30)
  LDC(a3, s0, -0x20)
#elif SL == 3
  STC(s0, t0, -0x30)
  LDC(a3, s0, 0x10)
#elif SL == 4
  STC(s0, t0, -0x30)
  LDC(a3, s1, -0x30)
#elif SL == 5
  sd   t2, -0x30(s0)
  ld   a5, -0x30(s0)
#elif SL == 6
  sd   t2, -0x30(s0)
  ld   a5, -0x20(s0)
#elif SL == 7
  LDC(a3, s0, -0x30)
#elif SL == 8
  STC(s0, t0, -0x30)
#endif
  addi t2, t2, -1
  bnez t2, 3b

  csrr s7, mcycle
  csrr s8, minstret
  sub  s7, s7, s5
  sub  s8, s8, s6
#if SL <= 4 || SL == 7
  lbu  a5, 0(a3)
#endif
  CAPPRINT(s7)                   # cycles
  CAPPRINT(s8)                   # instructions (control)
  CAPPRINT(a5)                   # value
  CAPPRINT(s10)                  # trap count: 0
  CAPPRINT(s4)                   # last cause: 0
  CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
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
.align 14
heap:
  .zero 0x4000
heap_end:
RVTEST_DATA_END
