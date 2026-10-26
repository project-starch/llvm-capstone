# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# fp-of-divsqrt.S  (2026-10-26: an overflowing fdiv under every rounding mode)
#-----------------------------------------------------------------------------
# R-62's residual list says the divsqrt unit returns infinity on overflow whatever the rounding mode. By reading
# (core/fpu_div_sqrt_fix/norm_div_sqrt_mvp.sv, the OF branch): Exp_res_norm_D = '1, Mant_res_norm_D = 0,
# Mant_forround_D = 0, so the rounding stage adds nothing and In_Exact_S = Mant_rounded_S = 0. IEEE 754 7.4: an
# overflow returns infinity under RNE and RMM, the largest finite magnitude under RTZ, and under RDN / RUP whichever
# of the two lies toward the rounding direction; it raises OF and NX together.
# This arm reads DBL_MAX / 0.5, -DBL_MAX / 0.5 and FLT_MAX / 0.5 under frm = 0..4 (15 readings), each followed by
# fflags (cleared before each op), so 30 prints.
# IEEE (per frm 0..4):
#   +DBL_MAX/0.5  7ff0000000000000 7fefffffffffffff 7fefffffffffffff 7ff0000000000000 7ff0000000000000
#   -DBL_MAX/0.5  fff0000000000000 ffefffffffffffff fff0000000000000 ffefffffffffffff fff0000000000000
#   +FLT_MAX/0.5  7f800000         7f7fffff         7f7fffff         7f800000         7f800000
#   fflags        0x5 after every op
# PREDICTED by reading on e7af48457: infinity of the result's sign in all 15, fflags 0x4 (OF without NX).
# READ 2026-10-26 on e7af48457 (.sup/fpof; fp-rm-divsqrt in the same model read 15/15 correct): as predicted, all 30
# prints: 7ff0000000000000 / fff0000000000000 / 7f800000 under every mode, fflags 0x4 after every op. So an overflowing
# division never raises NX, under RNE too, and returns infinity where IEEE gives the largest finite magnitude (RTZ;
# RDN for a positive result; RUP for a negative one). The branch is the vendored one, unchanged by R-62/R-63.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li t0, 0x6000
csrs mstatus, t0
csrw fcsr, x0
lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)
li t0, 0x7fefffffffffffff
fmv.d.x fa0, t0                # DBL_MAX
fneg.d fa2, fa0                # -DBL_MAX
li t0, 0x3fe0000000000000
fmv.d.x fa1, t0                # 0.5
li t0, 0x7f7fffff
fmv.w.x fa3, t0                # FLT_MAX
li t0, 0x3f000000
fmv.w.x fa4, t0                # 0.5f
li s0, 0                       # frm
1:
fsrm s0
csrw fflags, x0
fdiv.d ft0, fa0, fa1
csrr t2, fflags
fmv.x.d t1, ft0
CAPPRINT(t1)
CAPPRINT(t2)
csrw fflags, x0
fdiv.d ft0, fa2, fa1
csrr t2, fflags
fmv.x.d t1, ft0
CAPPRINT(t1)
CAPPRINT(t2)
csrw fflags, x0
fdiv.s ft0, fa3, fa4
csrr t2, fflags
fmv.x.w t1, ft0
CAPPRINT(t1)
CAPPRINT(t2)
addi s0, s0, 1
li t2, 5
bne s0, t2, 1b
CAP_PASS(s11)

.align 4
trap_handler:
  csrr t0, mcause
  CAPPRINT(t0)
  CAP_PASS(s11)

.align 4
_end_of_text:

RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
