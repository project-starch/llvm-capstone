# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# fp-carry-divsqrt.S  (2026-10-26: a subnormal fdiv result that rounds up into the smallest normal)
#-----------------------------------------------------------------------------
# The subnormal-result branches of the normaliser leave the hidden-bit position (bit 52) at 0, and the
# renormalisation tests only bit 53 (Mant_renorm_S = Mant_upperRounded_D[C_MANT_FP64+1]) and adds only that to the
# exponent. A round-up that carries into bit 52 is the step to the smallest normal; by reading, it is dropped and
# the result reads zero of the result's sign. Found by the R-64 fix's claim-auditor on a normaliser-only harness.
# Four divisions under frm = 0..4, each followed by fflags (cleared before each op), so 40 prints:
#   A  0x001fffffffffffff / 2.0   = (1 - 2^-53) * 2^-1022, a tie between the largest subnormal and the smallest normal
#   B  0x001ffffffffffffe / 2.0   = the largest subnormal, exact (the no-carry neighbour)
#   C  0x00ffffff / 2.0f          = the FP32 twin of A
#   D  -A
# IEEE (host FPU under fenv, x86-64; RMM from its definition: ties away = the RNE value here):
#           RNE                 RTZ                 RDN                 RUP                 RMM
#   A  0010000000000000 3  000fffffffffffff 3  000fffffffffffff 3  0010000000000000 3  0010000000000000 3
#   B  000fffffffffffff 0  000fffffffffffff 0  000fffffffffffff 0  000fffffffffffff 0  000fffffffffffff 0
#   C  00800000 3          007fffff 3          007fffff 3          00800000 3          00800000 3
#   D  8010000000000000 3  800fffffffffffff 3  8010000000000000 3  800fffffffffffff 3  8010000000000000 3
# PREDICTED on e7af48457 (by reading, written before the run): every IEEE 0010.../8010.../00800000 entry reads zero
# of that sign (0x0, 0x8000000000000000, 0x0), flags 0x3; the other entries read IEEE; B reads fflags 0x2 (R-64's
# exact-subnormal UF).
# READ 2026-10-26 on the e7af48457 core (r62-divsqrt 284de8f0f + this test, .sup/fpcarry): all 40 prints as PREDICTED.
# The nine carry entries (A and C under RNE/RUP/RMM, D under RNE/RDN/RMM) read 0x0 / 0x8000000000000000 / 0x0 with fflags 0x3; every other entry reads IEEE; B reads 0x2.
# fp-rm-divsqrt in the same model read 15/15.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define OPD(...) csrw fflags, x0; __VA_ARGS__; csrr t2, fflags; fmv.x.d t1, ft0; CAPPRINT(t1); CAPPRINT(t2)
#define OPS(...) csrw fflags, x0; __VA_ARGS__; csrr t2, fflags; fmv.x.w t1, ft0; CAPPRINT(t1); CAPPRINT(t2)

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
li t0, 0x001fffffffffffff
fmv.d.x fs0, t0
li t0, 0x001ffffffffffffe
fmv.d.x fs1, t0
li t0, 0x4000000000000000
fmv.d.x fs2, t0                # 2.0
fneg.d fs3, fs0                # -A's dividend
li t0, 0x00ffffff
fmv.w.x fs4, t0
li t0, 0x40000000
fmv.w.x fs5, t0                # 2.0f
li s0, 0
1:
fsrm s0
OPD(fdiv.d ft0, fs0, fs2)
OPD(fdiv.d ft0, fs1, fs2)
OPS(fdiv.s ft0, fs4, fs5)
OPD(fdiv.d ft0, fs3, fs2)
addi s0, s0, 1
li t3, 5
bne s0, t3, 1b
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
