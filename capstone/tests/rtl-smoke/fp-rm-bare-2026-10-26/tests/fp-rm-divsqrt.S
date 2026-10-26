# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# fp-rm-divsqrt.S  (2026-10-26: fdiv.d / fsqrt.d under every rounding mode)
#-----------------------------------------------------------------------------
# fpnew encodes the rounding modes RNE 0, RTZ 1, RDN 2, RUP 3, RMM 4 (fpnew_pkg.sv:131-135). The divsqrt unit takes
# rnd_mode_i unmapped (fpnew_divsqrt_multi.sv:101, :285), and its own constants are NEAREST 0, TRUNC 1, PLUSINF 2,
# MINUSINF 3 (defs_div_sqrt_mvp.sv:18-21). So by reading, RDN rounds toward +inf and RUP toward -inf, and RMM (4) falls
# to the default, which never rounds up. This arm reads it: 1/3, -1/3 and sqrt(2) under frm = 0..4, in that order
# (15 readings), then fflags.
# Correctly rounded (Python, exact rationals):
#   1/3    RNE 0x3fd5555555555555 RTZ ...555 RDN ...555 RUP ...556 RMM ...555
#   -1/3   RNE 0xbfd5555555555555 RTZ ...555 RDN ...556 RUP ...555 RMM ...555
#   sqrt2  RNE 0x3ff6a09e667f3bcd RTZ ...bcc RDN ...bcc RUP ...bcd RMM ...bcd
# Read 2026-10-26 on the R-62 fix a73a7c670 (rounding modes still unmapped): RNE and RTZ correct; RDN and RUP exchanged
# value for value; RMM truncating (sqrt2 ...bcc). With the mode decoded in fpnew's encoding (the R-62 normaliser), all 15
# are predicted correctly rounded: none of these results has all-zero extra bits, which is where the directed modes
# still need the remainder.

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
li t0, 1
fcvt.d.l fa0, t0               # 1.0
li t0, 3
fcvt.d.l fa1, t0               # 3.0
fneg.d fa2, fa0                # -1.0
li t0, 2
fcvt.d.l fa3, t0               # 2.0
li s0, 0                       # frm
1:
fsrm s0
fdiv.d ft0, fa0, fa1
fmv.x.d t1, ft0
CAPPRINT(t1)
fdiv.d ft0, fa2, fa1
fmv.x.d t1, ft0
CAPPRINT(t1)
fsqrt.d ft0, fa3
fmv.x.d t1, ft0
CAPPRINT(t1)
addi s0, s0, 1
li t2, 5
bne s0, t2, 1b
csrr t1, fflags
CAPPRINT(t1)
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
