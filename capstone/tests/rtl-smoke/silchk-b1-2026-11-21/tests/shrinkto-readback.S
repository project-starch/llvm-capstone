# shrinkto-readback.S -- what size SHRINKTO imm produces (R-32), read back as integers
#
# R-32: before the fix (capstone-ariane a3b7a22c0) SHRINKTO sets `rd_end = cursor + imm - 1` in an RTL whose `end`
# is EXCLUSIVE, so the child holds imm - 1 bytes; with the fix it holds imm. QEMU (helper_csshrinkto) holds imm.
# A pass of compiled code cannot tell imm - 1 from imm when the compiler adds 1 to the immediate, so this reads the
# bounds directly: every arm builds a fresh LINEAR capability over the 4096-byte `region`, narrows it, and records
# LCC field 3 (start) and field 4 (end), each minus `region`'s address, through the board recorder (CAPPRINT of an
# integer). Nothing is stored or loaded by the test itself.
#
# Vector, in print order (fixed RTL / pre-fix RTL):
#   0,1   the parent: start 0, end 4096 / the same (LCC's end is exclusive on both)
#   2,3   SHRINK control, bounds [region, region+64) given as integers: 0, 64 / the same
#   4,5   SHRINKTO 64, cursor at start:            0, 64   / 0, 63
#   6,7   SHRINKTO 1:                               0, 1    / 0, 0
#   8,9   SHRINKTO 17:                              0, 17   / 0, 16
#   10,11 SHRINKTO 2047 (the largest immediate):    0, 2047 / 0, 2046
#   12,13 cursor at 4032, SHRINKTO 64 (exact fit, cursor + imm == end): 4032, 4096 / 4032, 4095
#   14,15 cursor at 16, SHRINKTO 32:                16, 48  / 16, 47
# A trap prints mcause as one more value and ends the run (trap_handler).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define MKCAP(c) \
    CAPCREATE(c); li t0, CAP_TYPE_LIN; CAPTYPE(c, t0); \
    lla t1, region; li t2, 4096; add t2, t1, t2; CAPBOUND(c, t1, t2); \
    li t0, CAP_PERM_RWX; CAPPERM(c, t0)

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
lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

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
.align 12
region:
  .fill 512, 8, 0
  TEST_DATA
RVTEST_DATA_END
