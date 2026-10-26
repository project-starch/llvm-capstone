# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# fp-special-divsqrt.S  (2026-10-26: fdiv.d / fsqrt.d on special operands, result and fflags; RNE)
#-----------------------------------------------------------------------------
# Eleven ops, each preceded by csrw fflags, x0 and followed by two prints, the result bits and fflags (22 prints):
#   op                 IEEE / RISC-V result   fflags   PREDICTED by reading core/fpu_div_sqrt_fix/norm_div_sqrt_mvp.sv
#   1  inf / 2         7ff0000000000000       0x0      0x4  (the Inf_a branch sets Exp_OF_S)
#   2  2 / inf         0                      0x0      0x4  (the Inf_b branch sets Exp_OF_S)
#   3  sqrt(inf)       7ff0000000000000       0x0      0x4  (the Inf_a branch)
#   4  0 / 0           7ff8000000000000       0x10     0x18 (Div_Zero_S and NV_OP_S both set)
#   5  1 / 0           7ff0000000000000       0x8      0x8
#   6  -1 / 0          fff0000000000000       0x8      0x8
#   7  inf / inf       7ff8000000000000       0x10     0x10
#   8  sqrt(-1)        7ff8000000000000       0x10     0x10
#   9  0 / 2           0                      0x0      0x0
#   10 sqrt(-0)        8000000000000000       0x0      0x0
#   11 4*min_sub / 2   0000000000000002       0x0      0x2  (an exact subnormal result: the denormal branch sets
#                                                            Exp_UF_S; RISC-V raises UF only when tiny AND inexact)
# Results are predicted to equal IEEE in every row; only the flags differ.
# READ 2026-10-26 on e7af48457 (.sup/fpsp): all 22 prints as PREDICTED. Results equal IEEE in all eleven rows; fflags
# wrong in rows 1-4 and 11, right in rows 5-10 (the control that the flag readout works). The IEEE column was checked
# against the host FPU under fenv (x86-64, glibc; its default NaN is negative, RISC-V's canonical NaN is 7ff8...).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define OPD(...) csrw fflags, x0; __VA_ARGS__; csrr t2, fflags; fmv.x.d t1, ft0; CAPPRINT(t1); CAPPRINT(t2)

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
li t0, 0x7ff0000000000000
fmv.d.x fs0, t0                # inf
li t0, 0x4000000000000000
fmv.d.x fs1, t0                # 2.0
fmv.d.x fs2, x0                # 0.0
li t0, 0x3ff0000000000000
fmv.d.x fs3, t0                # 1.0
fneg.d fs4, fs3                # -1.0
fneg.d fs5, fs2                # -0.0
li t0, 4
fmv.d.x fs6, t0                # 4 * smallest subnormal
OPD(fdiv.d ft0, fs0, fs1)
OPD(fdiv.d ft0, fs1, fs0)
OPD(fsqrt.d ft0, fs0)
OPD(fdiv.d ft0, fs2, fs2)
OPD(fdiv.d ft0, fs3, fs2)
OPD(fdiv.d ft0, fs4, fs2)
OPD(fdiv.d ft0, fs0, fs0)
OPD(fsqrt.d ft0, fs4)
OPD(fdiv.d ft0, fs2, fs1)
OPD(fsqrt.d ft0, fs5)
OPD(fdiv.d ft0, fs6, fs1)
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
