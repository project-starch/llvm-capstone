# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-r32-shrinkto.S -- ISSUES R-32 on Line B: SHRINKTO imm gives a region of imm bytes, read back as integers (2026-11-26)
#-----------------------------------------------------------------------------
#
# The Line B port of P1's shrinkto-readback.S (silchk-b1-2026-11-21 / r32-shrinkto-bare-2026-11-06), which needs
# CAPENTER and the custom-3 mints, both deleted under the Sublet format. Here every arm takes a fresh capability over
# the 4096-byte `region` by CSMINT (in M, the physical pool; a Sublet capability decodes NONLIN with every right, which
# SHRINKTO accepts), narrows it, and records LCC field 3 (start) and field 4 (end), each minus region's address. M is
# integer mode on Line B, and LCC / SHRINK / SHRINKTO / CINCOFFSET decode there (decoder.sv, no Sublet illegal term).
# Vector, in print order: the fixed RTL / the R-32 fix reverted (`let rd_end = inc - 64'd1;`, a local sim-only patch):
#   0,1   the parent: 0, 4096 / the same
#   2,3   SHRINK control [region, region+64): 0, 64 / the same
#   4,5   SHRINKTO 64:                          0, 64   / 0, 63
#   6,7   SHRINKTO 1:                           0, 1    / 0, 0
#   8,9   SHRINKTO 17:                          0, 17   / 0, 16
#   10,11 SHRINKTO 2047:                        0, 2047 / 0, 2046
#   12,13 cursor 4032, SHRINKTO 64 (exact fit): 4032, 4096 / 4032, 4095
#   14,15 cursor 16, SHRINKTO 32:               16, 48  / 16, 47
# A trap prints mcause as one more value and ends the run.
# PRE-REGISTERED (2026-11-26, written before any run), on b42c8c644's core (capstone-ariane b5b2-hangmon-c + this test):
#   the fixed column on the tree, the reverted column on the negative control. P1's vectors, unchanged by the port.
# FIRST RUN (449460a34, 2026-11-26): AS PREDICTED, both columns exactly. Fixed (anvil_flu_unit 945d64818c37e6c9):
#   0x0 0x1000 0x0 0x40 0x0 0x40 0x0 0x1 0x0 0x11 0x0 0x7ff 0xfc0 0x1000 0x10 0x30. The fix reverted (a local sim
#   patch, diff sha256 087221188ba1fe4f..., anvil_flu_unit ed3bb31ae3dc1050, so the unit was regenerated):
#   0x0 0x1000 0x0 0x40 0x0 0x3f 0x0 0x0 0x0 0x10 0x0 0x7fe 0xfc0 0xfff 0x10 0x2f.
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CS_MINT_INTO(REG, LO, HI) \
  .pushsection .data; .balign 16; 77: .zero 64; .popsection; \
  lla a0, 77b; lla a3, LO; sd a3, 0(a0); lla a4, HI; sd a4, 8(a0); li a5, 0x10f; sd a5, 16(a0); \
  sd a3, 24(a0); sd x0, 32(a0); sd x0, 40(a0); addi a5, a0, 48; .insn r 0x5B, 0x1, 0x50, a3, a5, a0; LDC(REG, a5, 0)
#define MKCAP(c) CS_MINT_INTO(c, region, region_end)
#define READ_BOUNDS(c) \
    LCC(a2, c, x3); LCC(a3, c, x4); lla t1, region; sub a2, a2, t1; sub a3, a3, t1; \
    CAPPRINT(a2); CAPPRINT(a3)

#define SHRINKTO(c, imm) .insn i 0x5b, 0x0, c, c, imm

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0

# 0,1: the parent
MKCAP(a1)
READ_BOUNDS(a1)

# 2,3: SHRINK control (integer bounds, the same on both trees)
MKCAP(a1)
lla t1, region
addi t2, t1, 64
SHRINK(a1, t1, t2)
READ_BOUNDS(a1)

# 4..11: SHRINKTO at the start
MKCAP(a1)
SHRINKTO(a1, 64)
READ_BOUNDS(a1)
MKCAP(a1)
SHRINKTO(a1, 1)
READ_BOUNDS(a1)
MKCAP(a1)
SHRINKTO(a1, 17)
READ_BOUNDS(a1)
MKCAP(a1)
SHRINKTO(a1, 2047)
READ_BOUNDS(a1)

# 12,13: exact fit at the parent's end
MKCAP(a1)
li t0, 4032
CINCOFFSET(a1, a1, t0)
SHRINKTO(a1, 64)
READ_BOUNDS(a1)

# 14,15: cursor moved off the start
MKCAP(a1)
li t0, 16
CINCOFFSET(a1, a1, t0)
SHRINKTO(a1, 32)
READ_BOUNDS(a1)

RVTEST_PASS

.align 4
trap_handler:
  csrr t0, mcause
  CAPPRINT(t0)
  RVTEST_PASS

.align 4
_end_of_text:

RVTEST_CODE_END
  .data
RVTEST_DATA_BEGIN
.align 12
region:
  .fill 512, 8, 0
region_end:
  TEST_DATA
RVTEST_DATA_END
