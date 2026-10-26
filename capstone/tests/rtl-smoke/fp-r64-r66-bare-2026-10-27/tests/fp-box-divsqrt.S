# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# fp-box-divsqrt.S  (2026-10-26: fdiv.s / fsqrt.s on an FP32 operand that is not NaN-boxed)
#-----------------------------------------------------------------------------
# RISC-V D (NaN boxing): an FP32 operation that reads an f register whose upper 32 bits are not all ones treats the
# operand as the canonical NaN. fpnew computes is_boxed per operand and hands it to every unit; the FMA unit passes it
# to its classifier (fpnew_fma_multi.sv:130, :196), but fpnew_divsqrt_multi.sv declares is_boxed_i (:33) and never
# reads it, so by reading the divide/sqrt unit computes on the low 32 bits as if they were boxed.
# Five ops, each followed by fflags (cleared before each op), 10 prints:
#   1 fdiv.s  unboxed 1.0f / boxed 2.0f     RISC-V: ffffffff7fc00000 0x0   PREDICTED: ffffffff3f000000 0x0 (0.5f)
#   2 fdiv.s  boxed 1.0f / unboxed 2.0f     RISC-V: ffffffff7fc00000 0x0   PREDICTED: ffffffff3f000000 0x0
#   3 fsqrt.s unboxed 4.0f                  RISC-V: ffffffff7fc00000 0x0   PREDICTED: ffffffff40000000 0x0 (2.0f)
#   4 fadd.s  unboxed 1.0f + boxed 2.0f     RISC-V: ffffffff7fc00000 0x0   PREDICTED: as RISC-V (the FMA unit checks)
#   5 fdiv.s  boxed 1.0f / boxed 2.0f       RISC-V: ffffffff3f000000 0x0   (control: a boxed division is correct)
# "Unboxed" here is the FP32 pattern in the low half with zeros above (fmv.d.x of 0x000000003f800000).
# READ 2026-10-26 on the e7af48457 core (r62-divsqrt 4ba8d64e8 + this test, .sup/fpbox): all 10 prints as PREDICTED: rows
# 1-3 compute on the unboxed low half (3f000000, 3f000000, 40000000), row 4 (FMA unit) reads the canonical NaN, row 5
# (boxed control) is correct; fflags 0x0 throughout.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define OPS(...) csrw fflags, x0; __VA_ARGS__; csrr t2, fflags; fmv.x.d t1, ft0; CAPPRINT(t1); CAPPRINT(t2)

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
li t0, 0x3f800000
fmv.d.x fs0, t0                # unboxed 1.0f
fmv.w.x fs1, t0                # boxed 1.0f
li t0, 0x40000000
fmv.d.x fs2, t0                # unboxed 2.0f
fmv.w.x fs3, t0                # boxed 2.0f
li t0, 0x40800000
fmv.d.x fs4, t0                # unboxed 4.0f
OPS(fdiv.s ft0, fs0, fs3)
OPS(fdiv.s ft0, fs1, fs2)
OPS(fsqrt.s ft0, fs4)
OPS(fadd.s ft0, fs0, fs3)
OPS(fdiv.s ft0, fs1, fs3)
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
