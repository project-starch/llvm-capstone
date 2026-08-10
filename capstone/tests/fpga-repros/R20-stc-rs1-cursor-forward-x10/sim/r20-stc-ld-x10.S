# See LICENSE for license details.

#*****************************************************************************
# r20-stc-ld-x10.S
#-----------------------------------------------------------------------------
#
# Directed test for R-20, observed on silicon (caplifive_65536_r18_fix.bit) and packaged at
# project-starch/llvm-capstone capstone/tests/fpga-repros/R20-stc-rs1-cursor-forward-x10/.
#
# BOARD OBSERVATION. A capability store to an address, immediately followed by a plain integer
# load from that same address into the SAME register, immediately followed by any consumer of
# that register, gives the consumer the ADDRESS THE STORE USED instead of the loaded value --
# but only when the register is x10/a0. The same sequence on another register is correct, a
# plain sd instead of stc is correct, and one nop on either side cures it.
#
# WHY A SIMULATION IS WORTH RUNNING. On the board this is inferred from branch direction, which
# needs a model of the branch to interpret. Here the RVFI trace prints the value each
# instruction writes, so the consumer's operand is read DIRECTLY. The three arms below differ
# only in the register they use and in one nop, so the trace answers the question outright.
#
# WHAT TO LOOK FOR in the RVFI trace, on the `addi` that follows each load:
#   value 0                      -> correct: the consumer saw the loaded NULL
#   value == test_buf_start      -> R-20 reproduced: it saw the store's base address instead
#
# The stored value is a NULL capability, made with MOVC from x0, so the correct loaded value is
# unambiguously 0 and any nonzero answer is the defect rather than a legitimate pointer.
#
# Arms:
#   A  x10/a0, consumer adjacent        expect the defect if it reproduces here
#   B  x13/a3, consumer adjacent        CONTROL: different register, must read 0
#   C  x10/a0, one nop before consumer  CONTROL: separation, must read 0
#
# Arms B and C are the instrument validation. If either reads nonzero, this test cannot tell a
# good case from a bad one and the run carries no verdict.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc          # match the board's 4-byte encodings; the assembler otherwise
                       # emits c.ld/c.mv/c.nop and compression becomes a hidden variable
.section .text.init
.globl _start
_start:

# NOTES, each of which cost a run:
#  * CHK_START / CHK_END expand to `addi x0, a0, 0` and `addi x0, a1, 0` -- they READ a0/a1 as
#    plain integers, and at these points those hold live capabilities, which raises
#    UNEXPECTED_OPERAND on this core. Same for CAPPRINT of an integer register. Not used here.
#  * A LINEAR capability MOVES rather than copies, so one parked capability cannot serve three
#    arms -- the first arm consumes it and the next arm's STC gets a non-capability address.
#  * The helper registers must never themselves have held a capability: an ordinary instruction
#    does not clear a register's capability metadata, so a stale tag turns a plain-integer
#    operand into UNEXPECTED_OPERAND. a0/a2/a3/a4 are used ONLY as helpers below, and the three
#    capabilities are built up front, before any arm runs.

#define MKCAP(CREG)          \
  CAPCREATE(CREG);           \
  li  a0, CAP_TYPE_LIN;      \
  lla a2, test_buf_start;    \
  lla a3, test_buf_end;      \
  li  a4, CAP_PERM_RWX;      \
  CAPTYPE(CREG, a0);         \
  CAPBOUND(CREG, a2, a3);    \
  CAPPERM(CREG, a4);

MKCAP(a5)
MKCAP(a6)
MKCAP(a7)

# ---------------------------------------------------------------- ARM A: x10/a0, adjacent
MOVC(a0, a5)
MOVC(a1, x0)            # a1 = NULL capability
STC(a0, a1, 0)          # stc a1, 0(a0)
ld   a0, 0(a0)          # the load's OWN retired value is visible in the trace, and is 0
addi t3, a0, 0          # CONSUMER -- RVFI prints what it actually read

# ---------------------------------------------------------------- ARM B: x13/a3, adjacent
MOVC(a3, a6)
MOVC(a1, x0)
STC(a3, a1, 0)
ld   a3, 0(a3)
addi t4, a3, 0          # CONSUMER -- CONTROL, different register, must read 0

# ------------------------------------------------- ARM C/D/E: x10/a0, separation sweep
MOVC(a0, a7)
MOVC(a1, x0)
STC(a0, a1, 0)
ld   a0, 0(a0)
nop
addi t5, a0, 0          # 1 nop

MKCAP(s3)
MOVC(a0, s3)
MOVC(a1, x0)
STC(a0, a1, 0)
ld   a0, 0(a0)
nop
nop
addi s4, a0, 0          # 2 nops

MKCAP(s5)
MOVC(a0, s5)
MOVC(a1, x0)
STC(a0, a1, 0)
ld   a0, 0(a0)
nop
nop
nop
nop
addi s6, a0, 0          # 4 nops

RVTEST_PASS

  .data
RVTEST_DATA_BEGIN

.align 4
test_buf_start:
  .zero 4096
test_buf_end:

.align 3
results:
  .dword 0
  .dword 0
  .dword 0

  TEST_DATA

RVTEST_DATA_END
