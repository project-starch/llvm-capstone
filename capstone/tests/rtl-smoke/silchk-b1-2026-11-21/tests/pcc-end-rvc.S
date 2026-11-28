# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# pcc-end-rvc.S -- does a 32-bit instruction whose upper half lies past the PC capability's end commit? (2026-11-13)
#-----------------------------------------------------------------------------
#
# commit_stage.sv pc_cap_check (M + capmode) tests only the start pc against [bound_start, bound_end), with no
# instruction length; QEMU's virtual-context path tests [pc, pc + len). With compressed code a 32-bit instruction can
# start at bound_end - 2 under 4-aligned bounds.
#
# CONSTRUCTION. s11 is a LINEAR RWX code capability over [pcarm, END), entered by cjalr; a trap does not change the PC
# capability, so the handler lies inside it. With compressed code the probe places `straddle`, a 32-bit addi of 0x123
# into s7, at 2 mod 4. The three arms differ ONLY in END:
#   default:  END = straddle + 2   (the instruction's upper half is past the end)
#   -DPE_IN:  END = straddle + 4   (control: the instruction lies wholly inside)
#   -DPE_OUT: END = straddle       (control: the instruction lies wholly outside)
# The handler records mcause, mepc - straddle and mtval - straddle, and ends the test.
# Prints: s7, mcause, mepc - straddle, mtval - straddle, trap count.
# PRE-REGISTERED READINGS (from the source, before any run):
#   default: 0x123 0x1c 0x4 0x4 0x1   (the straddler commits; the next fetch, at straddle + 4, faults 28)
#   PE_IN:   0x123 0x1c 0x4 0x4 0x1   (identical to the default: the leniency)
#   PE_OUT:  0x0   0x1c 0x0 0x0 0x1   (the straddler itself faults)
# A whole-instruction check, as QEMU's, would read the default as PE_OUT.
# READ 2026-11-13 on 6d0e754d5's RTL (vp9 .sup/pe1, rtl 4190b2ebd dirty=no), all three SUCCESS, every value as
# predicted: default 0x123 0x1c 0x4 0x4 0x1; PE_IN the same; PE_OUT 0x0 0x1c 0x0 0x0 0x1. The straddler commits.
# WITH THE R-72 FIX (commit_stage.sv: fewer than the instruction's length left before bound_end is out of bounds),
# predicted before its run: default 0x0 0x1c 0x0 0x0 0x1 (the straddler now faults at itself); PE_IN and PE_OUT
# unchanged (0x123 0x1c 0x4 0x4 0x1 and 0x0 0x1c 0x0 0x0 0x1).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#ifndef CAP_PASS
#define CAP_PASS(creg) \
    CAPCREATE(creg); li a0, CAP_TYPE_LIN; lla a2, tohost; lla a3, tohost + 64; li a4, CAP_PERM_RWX; \
    CAPTYPE(creg, a0); CAPBOUND(creg, a2, a3); CAPPERM(creg, a4); \
    fence; li a0, 1; sw a0, 0(creg); 1: j 1b
#endif

.option norvc
.section .text.init
.globl _start
_start:
  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s7, 0
  li   s10, 0
  li   s9, -1
  li   s8, -1
  li   s5, -1
  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)

  CAPCREATE(s11)
  li   t1, CAP_TYPE_LIN
  CAPTYPE(s11, t1)
  lla  a2, pcarm
  lla  a3, straddle
#if defined(PE_IN)
  addi a3, a3, 4
#elif defined(PE_OUT)
  addi a3, a3, 0
#else
  addi a3, a3, 2
#endif
  CAPBOUND(s11, a2, a3)
  li   t1, CAP_PERM_RWX
  CAPPERM(s11, t1)
  lla  t1, pcarm
  SCC(s11, s11, t1)
  CJALR(t6, s11, 0)

.align 6
pcarm:
  j    probe

.align 4
trap_handler:
  csrr s9, mcause
  csrr s8, mepc
  csrr s5, mtval
  addi s10, s10, 1
  lla  t3, straddle
  sub  s8, s8, t3
  sub  s5, s5, t3
  CAPPRINT(s7)
  CAPPRINT(s9)
  CAPPRINT(s8)
  CAPPRINT(s5)
  CAPPRINT(s10)
  CAP_PASS(s6)

.align 4
probe:
.option rvc
  li   s7, 0
straddle:
  addi s7, s7, 0x123
  nop
  nop
.option norvc

.align 4
_end_of_text:

RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
