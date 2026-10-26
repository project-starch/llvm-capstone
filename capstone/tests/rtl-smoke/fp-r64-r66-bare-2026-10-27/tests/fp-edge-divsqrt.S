# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# fp-edge-divsqrt.S  (2026-10-27: the R-65 / R-66 paths the fix's claim-auditor found no RTL arm for)
#-----------------------------------------------------------------------------
# Part 1, R-65 carries through the OTHER subnormal branch (Exp_in == 1, quotient significand below 1) and the FP32
# negative carries. Five divisions under frm = 0..4, each followed by fflags (cleared before each op), 50 prints:
#   P  0x001ffffffffffffe / 0x3fffffffffffffff   q = (2-2^-51)/(2-2^-52) < 1, Exp_in 1: just below half an ulp under
#                                                the largest subnormal's successor, with nonzero computed extra bits
#   N  -P's dividend / same
#   Q  0x00fffffe / 0x3fffffff (FP32 twin of P)
#   M  -Q's dividend / same
#   C  -0x00ffffff / 2.0f (the FP32 negative tie)
# IEEE (host FPU under fenv; RMM from its definition):
#        RNE                 RTZ                 RDN                 RUP                 RMM
#   P  000fffffffffffff 3  000fffffffffffff 3  000fffffffffffff 3  0010000000000000 3  000fffffffffffff 3
#   N  800fffffffffffff 3  800fffffffffffff 3  8010000000000000 3  800fffffffffffff 3  800fffffffffffff 3
#   Q  007fffff 3          007fffff 3          007fffff 3          00800000 3          007fffff 3
#   M  807fffff 3          807fffff 3          80800000 3          807fffff 3          807fffff 3
#   C  80800000 3          807fffff 3          80800000 3          807fffff 3          80800000 3
# Seven entries carry (P RUP, N RDN, Q RUP, M RDN, C RNE/RDN/RMM).
# Part 2, R-66 under RNE, 6 prints:
#   unboxed 1.0f / boxed sNaN 0x7fa00000          RISC-V ffffffff7fc00000 0x10
#   unboxed 1.0f / unboxed 2.0f                   RISC-V ffffffff7fc00000 0x0
#   0xffff00003f800000 (partly ones) / boxed 2.0f  RISC-V ffffffff7fc00000 0x0
# PREDICTED (written before any run):
#   on the e7af48457 core: the seven carry entries read zero of their sign (0x0 / 0x8000000000000000 / ffffffff00000000 /
#     ffffffff80000000), fflags 0x3; every other part-1 entry IEEE; part 2: canonical NaN 0x10, then 0.5f (3f000000) 0x0
#     twice;
#   on the r64-flags core: all 56 prints as the RISC-V column.
# READ 2026-10-27 on the e7af48457 core (r62-divsqrt c94148507, .sup/fpedge, dirty=no; prediction pushed first, at
# c94148507): all 56 prints as PREDICTED. The seven carries read zero of their sign, including P/Q through the Exp_in == 1
# branch; every other part-1 entry reads IEEE; part 2 reads canonical NaN 0x10, then 3f000000 0x0 twice.
# READ 2026-10-27 on the r64-flags core (76eeec974 + this test, capstone-ariane r64-flags-edge 319301b38, .sup/fpedgefix in
# vp9, dirty=no): all 56 prints as the RISC-V column, as PREDICTED.

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
li t0, 0x001ffffffffffffe
fmv.d.x fs0, t0
fneg.d fs1, fs0
li t0, 0x3fffffffffffffff
fmv.d.x fs2, t0
li t0, 0x00fffffe
fmv.w.x fs3, t0
fneg.s fs4, fs3
li t0, 0x3fffffff
fmv.w.x fs5, t0
li t0, 0x80ffffff
fmv.w.x fs6, t0
li t0, 0x40000000
fmv.w.x fs7, t0                # boxed 2.0f
li s0, 0
1:
fsrm s0
OPD(fdiv.d ft0, fs0, fs2)
OPD(fdiv.d ft0, fs1, fs2)
OPD(fdiv.s ft0, fs3, fs5)
OPD(fdiv.s ft0, fs4, fs5)
OPD(fdiv.s ft0, fs6, fs7)
addi s0, s0, 1
li t3, 5
bne s0, t3, 1b
fsrm x0
li t0, 0x3f800000
fmv.d.x fs8, t0                # unboxed 1.0f
li t0, 0x7fa00000
fmv.w.x fs9, t0                # boxed sNaN
li t0, 0x40000000
fmv.d.x fs10, t0               # unboxed 2.0f
li t0, 0xffff00003f800000
fmv.d.x fs11, t0               # upper half partly ones
OPD(fdiv.s ft0, fs8, fs9)
OPD(fdiv.s ft0, fs8, fs10)
OPD(fdiv.s ft0, fs11, fs7)
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
