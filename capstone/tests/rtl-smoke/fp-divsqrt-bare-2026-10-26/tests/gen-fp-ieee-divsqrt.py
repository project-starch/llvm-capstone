import random, struct, math, sys
random.seed(20261026)
def bits(x): return struct.unpack("<Q",struct.pack("<d",x))[0]
cases=[]
while len(cases)<512:
    a=random.uniform(1,2)*2.0**random.randint(-30,30)*random.choice([1,-1])
    b=random.uniform(1,2)*2.0**random.randint(-30,30)*random.choice([1,-1])
    cases.append((a,b))
rows="\n".join(f"  .dword {bits(a):#018x}, {bits(b):#018x}, {bits(a/b):#018x}, {bits(math.sqrt(abs(a))):#018x}" for a,b in cases)
src=r'''# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# fp-ieee-divsqrt.S  (2026-10-26: are fdiv.d and fsqrt.d correctly rounded on this core?)
#-----------------------------------------------------------------------------
# Why: ludcmp and st (BEEBS) read VERIFY-FAILED on bitstream 2 inside a context and pass on QEMU, whose softfloat is
# correctly rounded; both lean on the divide/sqrt unit (fpnew 0.7.0, divsqrt MERGED = fpu_div_sqrt_mvp). The context
# FP save/restore arms (va-runv-fp*) compare the FPU against itself and cannot see a consistent rounding error.
# 512 cases, seed 20261026: a, b = +-[1,2) x 2^[-30,30]; expected a/b and sqrt(|a|) from Python's IEEE doubles
# (round to nearest even, correctly rounded). M-mode, capmode, frm 0 (RNE). For each case fdiv.d and fsqrt.d are
# compared bit for bit against the table.
# Readings (CAPPRINT, in order): fdiv mismatches; fsqrt mismatches; the first fdiv mismatch: index, got, want;
# the first fsqrt mismatch: index, got, want (index -1 and zeros when none); the last case's fdiv bits (positive
# control that the loop ran to the end: the table's last a/b); fflags.
# Pre-registered prediction if the unit is IEEE-correct: 0 0 -1 0 0 -1 0 0 <last q> <NX set>.

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
CAPCREATE(s4)
li a0, CAP_TYPE_NONLIN
lla a3, tab
lla a4, tab_end
li a5, CAP_PERM_RW
CAPTYPE(s4, a0)
CAPBOUND(s4, a3, a4)
CAPPERM(s4, a5)
li s0, 0                       # fdiv mismatches
li s1, 0                       # fsqrt mismatches
li s2, -1                      # first fdiv index
li s3, 0
li s5, 0
li s6, -1                      # first fsqrt index
li s7, 0
li s8, 0
li s9, 0                       # index
li s10, NCASES
1:
fld fa0, 0(s4)
fld fa1, 8(s4)
ld t0, 16(s4)
ld t1, 24(s4)
fdiv.d fa2, fa0, fa1
fmv.x.d t2, fa2
mv a6, t2
beq t2, t0, 2f
addi s0, s0, 1
bgez s2, 2f
mv s2, s9
mv s3, t2
mv s5, t0
2:
fabs.d fa3, fa0
fsqrt.d fa3, fa3
fmv.x.d t2, fa3
beq t2, t1, 3f
addi s1, s1, 1
bgez s6, 3f
mv s6, s9
mv s7, t2
mv s8, t1
3:
CINCOFFSETIMM(s4, s4, 32)
addi s9, s9, 1
bne s9, s10, 1b
CAPPRINT(s0)
CAPPRINT(s1)
CAPPRINT(s2)
CAPPRINT(s3)
CAPPRINT(s5)
CAPPRINT(s6)
CAPPRINT(s7)
CAPPRINT(s8)
CAPPRINT(a6)
csrr t0, fflags
CAPPRINT(t0)
CAP_PASS(s11)

.align 4
trap_handler:
  csrr t0, mcause
  CAPPRINT(t0)
  CAP_PASS(s11)

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
tab:
'''+rows+r'''
tab_end:
  TEST_DATA
RVTEST_DATA_END
'''
src=src.replace("NCASES",str(len(cases)))
open(sys.argv[1],"w").write(src)   # usage: gen-fp-ieee-divsqrt.py <out.S>
print("last q", hex(bits(cases[-1][0]/cases[-1][1])))
