# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# sublet-life.S -- B5b-1 step 2a: the Sublet lifetime ops through the whole core, on the physical pool (2026-11-22)
#-----------------------------------------------------------------------------
#
# Integer M (no capenter, as sublet-fmt), so the lifetime ops use the physical pool, which the Sublet unit writes at
# reset as a flat recycling table: capacity 65,536, next 3, slot 2 = the debug mints' root (capcreate's default node 2,
# generation 0, depth 0). x26 = capcreate NONLIN RW over blk (256 bytes). Each operation's trap cause is printed (0 = no
# trap; the handler skips the instruction), then the pool's header and records 2..6 by plain loads.
#   c1  cderive x25 <- x26, 64          0
#   c2  cderive x24 <- x25, 32          0
#   c3  crevoke x25                     0     (ends x25 and its run, x24)
#   c4  cderive x20 <- x24, 32          25    (x24 is dead)
#   c5  cderive x23 <- x26, 64          0     (slot 3 again, generation 1)
#   c6  crenew x22 <- x23               0     (in place: generation 2)
#   c7  cderive x20 <- x23, 32          25    (x23 is stale)
#   c8  cderive x20 <- x22, 32          0     (a child of the renewed reference)
#   c9  crenew x22 <- x22               0     (rd = rs1; its child ends)
#   c10 crevoke x26                     0     (the root and everything below it)
#   c11 cderive x20 <- x26, size 0      25    (the dead parent is found before the zero size)
#   c12 crevoke x22                     25
#   c13 the two-operand crevoke encoding (rs2 = x25)   2  (illegal under the Sublet format)
#   c14 crenew with rs2 = x25           2
#   p1..p20  the pool's words at 0xbff00000 .. +0x98: header quadwords 0 and 1, records 2..6
# PRE-REGISTERED: sublet-life.expect, computed BEFORE any simulation by the reference's own table code (capstone-qemu
# 9289114e06 via sublet-life/build-harness.sh) on the same operation stream; the causes c13 and c14 from the decoder.
# Register map: cause s1 (x9), trap count a7, prints through t2 (the readings tool reads the .log column).

# B5b-2 (2026-11-24, the semantics owner's ruling): uses the capcreate family, to be ported to CSMINT. The family is
# illegal on the Line B Sublet bitstream once B5b-2 deletes it (last, after cjalr, ccsrrw and csrunv).
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CRENEW_S(rd, rs1)   .insn r 0x5B, 0x1, 0x54, rd, rs1, x0
#define CRENEW_BAD(rd, rs1) .insn r 0x5B, 0x1, 0x54, rd, rs1, x25
#define CHK  mv t2, s1; CAPPRINT(t2); li s1, 0

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
li a7, 0
li s1, 0

# x26: NONLIN RW over blk
CAPCREATE(x26)
li a0, CAP_TYPE_NONLIN
CAPTYPE(x26, a0)
lla a3, blk
lla a4, blk_end
CAPBOUND(x26, a3, a4)
li a5, CAP_PERM_RW
CAPPERM(x26, a5)
li a1, 64
li a2, 32
li a6, 0

CDERIVE(x25, x26, a1)
CHK
CDERIVE(x24, x25, a2)
CHK
CREVOKE(x25, x0)
CHK
CDERIVE(x20, x24, a2)
CHK
CDERIVE(x23, x26, a1)
CHK
CRENEW_S(x22, x23)
CHK
CDERIVE(x20, x23, a2)
CHK
CDERIVE(x20, x22, a2)
CHK
CRENEW_S(x22, x22)
CHK
CREVOKE(x26, x0)
CHK
CDERIVE(x20, x26, a6)
CHK
CREVOKE(x22, x0)
CHK
CREVOKE(x26, x25)
CHK
CRENEW_BAD(x20, x22)
CHK

# the pool, by plain loads
li t3, 0xbff00000
li t4, 20
1:
ld t2, 0(t3)
CAPPRINT(t2)
addi t3, t3, 8
addi t4, t4, -1
bnez t4, 1b

CAPPRINT(a7)
CAP_PASS(s10)

.align 4
trap_handler:
addi a7, a7, 1
csrr s1, mcause
csrr t6, mepc
addi t6, t6, 4
csrw mepc, t6
mret

.align 4
_end_of_text:

RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
  TEST_DATA
.align 8
blk:
.zero 256
blk_end:
RVTEST_DATA_END
