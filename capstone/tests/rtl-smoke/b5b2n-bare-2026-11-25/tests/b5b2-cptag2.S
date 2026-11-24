# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-cptag2.S -- B5b-2 step 6, part 2: CPTAG's acknowledge (the exact OR; an empty page's bit is cleared)
#-----------------------------------------------------------------------------
#
# The reference (capstone-qemu 9289114e06 cap_mem_map.c :504-522): a non-acknowledge answers the sticky may_hold_tags
# bit; an acknowledge first clears it if the page's actual tags are all zero, then answers the bit. The semantics
# owner's ruling (ii): the bitmap bit is that bit; an acknowledge on a marked page ORs the page's 256 tag bytes (33 tag
# words; 32 if the run is word-aligned), and if they are all zero clears the bit and invalidates the TWC (epoch bump).
# b5b2-sref.S's setup: x26/s10 = {0,3} live over data. a1 = pg_a, a2 = pg_b, never written before. A plain sd of a
# granule's first doubleword clears its tag. Readings are stored at x25/s9 and printed by M at the end, after their count.
# Type is LCC selector 1 (1 = a tagged NONLIN, 7 = untagged); an M ldc reads a granule's memory tag that way.
# Readings, in the semantics owner's order, one page (pg_a) unless stated:
#   1 stc x26 -> pg_a, then CPTAG mode 0
#   2 sd x0 -> pg_a (its only tag gone), then CPTAG mode 0 (sticky)       3 the type of an ldc of pg_a (the sanity check)
#   4 CPTAG mode 1 (the acknowledge)      5 CPTAG mode 0 after it
#   6 stc x26 -> pg_a again, then CPTAG mode 0      7 the type of an ldc of pg_a (the tag reads back through the refill)
#   8 / 9 the race: sd x0 -> pg_a, then CPTAG mode 1 and, back to back, stc x26 -> pg_a; then CPTAG mode 0
#   10 / 11 pg_b with one tag left: stc x26 -> pg_b and pg_b + 16, sd x0 -> pg_b, then CPTAG mode 1 and CPTAG mode 0
#   12 the trap count
# PRE-REGISTERED (2026-11-24, written before any run):
#   part 2:              1 | 1 7 | 0 0 | 1 1 | 0 1 | 1 1 | 0
#     and in the .iss: CPTAG-WALK lines for readings 4, 8 and 10 (results 0, 0, 1), CPTAG-CLEAR for 4 and 8 (pg_a)
#     only, and no other CPTAG-WALK
#   part 1 (b5b2-step6 eff52e152, the base: an acknowledge answers the sticky bit):
#                        1 | 1 7 | 1 1 | 1 1 | 1 1 | 1 1 | 0
# B5b-2 (2026-11-24, the semantics owner's ruling): uses the capcreate family, to be ported to CSMINT.

# LEGACY SCAFFOLD (B5b-2 step 9, 2026-11-25): this test mints with the custom-3 debug family, which decodes under the
# Sublet format only in simulation builds with CAPSTONE_SIM_DEBUG_MINT. -DCAP_MINT_CSMINT (the bare kits, silicon)
# mints through CSMINT instead, which allocates its own lifetime record, so pool ids shift by one there. New tests
# mint with CSMINT only.
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT    0x5C1

#ifdef CAP_MINT_CSMINT
/* CSMINT through a descriptor {base, end, perms 0x10f, cursor = base, 0, 0} and a granule private to this use (in
   .data), then an M ldc of the granule (M through a plain base keeps the tag). Clobbers a0, a3, a4, a5. */
#define CS_MINT_INTO(REG, LO, HI) \
  .pushsection .data; .balign 16; 77: .zero 64; .popsection; \
  lla a0, 77b; lla a3, LO; sd a3, 0(a0); lla a4, HI; sd a4, 8(a0); li a5, 0x10f; sd a5, 16(a0); \
  sd a3, 24(a0); sd x0, 32(a0); sd x0, 40(a0); addi a5, a0, 48; .insn r 0x5B, 0x1, 0x50, a3, a5, a0; LDC(REG, a5, 0)
/* the Sublet format stores no type and no rights: a loaded capability decodes NONLIN with every right */
#define MINT(REG, TYPE, LO, HI, PERM) CS_MINT_INTO(REG, LO, HI)
#else
#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)
#endif

#define CLR  li x9, 0

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
csrw medeleg, x0
csrw satp, x0
li x28, 0

# PMP entry 0: NAPOT over everything, RWX
li t0, -1
srli t0, t0, 10
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0

# ---------------------------------------------------------------- the table: sublet-ns.S's, PAGED, SL_F4's slot 4
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 3
slli t0, t0, 62
ori t0, t0, 512
sd t0, 0(s8)                   # capacity 512 | RECYCLE | PAGED
li t0, 6
sd t0, 8(s8)                   # next 6
sd x0, 16(s8)                  # no free list
li t0, 4
sd t0, 24(s8)                  # allocations
li t0, 0x32454741
slli t0, t0, 32
li t1, 0x50444f4e
or t0, t0, t1
sd t0, 32(s8)                  # magic NODPAGE2
sd x0, 40(s8)                  # retired 0
sd x0, 48(s8)                  # growth 0
sd x0, 56(s8)
lla t0, pg_d1
sd t0, 64(s8)                  # root entry 0 -> directory 1
lla t1, pg_d1
lla t0, pg_d2
sd t0, 0(t1)                   # directory 1 [0] -> directory 2
lla t1, pg_d2
lla t0, pg_rp0
sd t0, 0(t1)                   # directory 2 [0] -> record page 0
lla t0, pg_rp1
sd t0, 8(t1)                   # directory 2 [1] -> record page 1
li t0, -1
slli t0, t0, 32
lla t1, pg_rp0
li t2, 512
1:
sd t0, 8(t1)                   # the page-initialisation rule: every record's word 1 retired, both pages
addi t1, t1, 16
addi t2, t2, -1
bnez t2, 1b
lla t1, pg_rp0
li t0, 5
slli t0, t0, 34
sd t0, 32(t1)                  # slot 2: next 5, depth 0
sd x0, 40(t1)                  #         generation 0, prev 0
li t0, 4
slli t0, t0, 34
ori t0, t0, 8
sd t0, 80(t1)                  # slot 5: next 4, depth 1
li t0, 4
sd t0, 88(t1)                  #         prev 2
li t0, 3
slli t0, t0, 34
ori t0, t0, 8
sd t0, 64(t1)                  # slot 4: next 3, depth 1
li t0, 10
li t2, 1
slli t2, t2, 62
or t0, t0, t2
sd t0, 72(t1)                  #         prev 5, generation 2^30 (stale for the generation-0 id x27 carries)
li t0, 8
sd t0, 48(t1)                  # slot 3: next 0, depth 1
li t0, 8
sd t0, 56(t1)                  #         prev 4

# ---------------------------------------------------------------- the capabilities, by cderive in the physical pool
MINT(x18, CAP_TYPE_NONLIN, _start, tbl_end, CAP_PERM_RWX)  # pool node 2: {0,2}
lla a3, data
lla a4, _start
sub a3, a3, a4
CINCOFFSET(t5, x18, a3)
li a1, 4096
CDERIVE(x26, t5, a1)           # pool slot 3: {0,3} over data
CDERIVE(x27, t5, a1)           # pool slot 4: {0,4} over data
li t0, 0x77
sd t0, 0(x26)
sd x0, 8(x26)
lla x24, data                  # the integer address
li t0, 4088
CINCOFFSET(x20, x26, t0)       # x20/s4 = x26's cursor + 4088: ld 8(x20) reads data + 4096, one past the bounds
CINCOFFSET(x21, x27, t0)       # x21/s5 = x27's, the same

lla x29, refbox
li x21, 0
li x22, 0
li x23, 0
li x25, 0
lla x25, res                   # the reading pointer (x25/s9): M prints them all at the end
li x28, 0                      # the trap count, after every setup block

#define CPTAG(rd, rs1, mode) .insn i 0x0b, 6, rd, rs1, mode
#define REC       sd a5, 0(x25); addi x25, x25, 8
#define RES(R)    mv a5, R; REC
#define TY(R)     li a5, 0; LCC(a5, R, x1); REC

lla a1, pg_a
lla a2, pg_b
csrw SREVROOT, x0

# 1
STC(a1, x26, 0)
CPTAG(t1, a1, 0); RES(t1)
# 2, 3
sd x0, 0(a1)
CPTAG(t1, a1, 0); RES(t1)
li x22, 0; LDC(x22, a1, 0); TY(x22)
# 4, 5
CPTAG(t1, a1, 1); RES(t1)
CPTAG(t1, a1, 0); RES(t1)
# 6, 7
STC(a1, x26, 0)
CPTAG(t1, a1, 0); RES(t1)
li x22, 0; LDC(x22, a1, 0); TY(x22)
# 8, 9: the race
sd x0, 0(a1)
CPTAG(t1, a1, 1)
STC(a1, x26, 0)
CPTAG(t2, a1, 0)
RES(t1); RES(t2)
# 10, 11: one tag left
STC(a2, x26, 0)
STC(a2, x26, 16)
sd x0, 0(a2)
CPTAG(t1, a2, 1); RES(t1)
CPTAG(t1, a2, 0); RES(t1)
mv a5, x28; REC

# ---------------------------------------------------------------- print: the count, then every reading
lla t1, res
sub t2, x25, t1
srli t2, t2, 3
CAPPRINT(t2)
1:
ld a5, 0(t1)
CAPPRINT(a5)
addi t1, t1, 8
bne t1, x25, 1b
RVTEST_PASS

.align 4
trap_handler:
  csrr x9, mcause
  li t0, 8
  beq x9, t0, 1f
  li t0, 9
  beq x9, t0, 1f
  addi x28, x28, 1
  csrr x19, mtval
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
1:
  jr x30

.align 12
img_end:
RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
data:
  .zero 8192
.align 12
tbl:
  .zero 4096
pg_d1:
  .zero 4096
pg_d2:
  .zero 4096
pg_rp0:
  .zero 4096
pg_rp1:
  .zero 4096
tbl_end:
.align 12
pg_a:
  .zero 4096
pg_b:
  .zero 4096
.align 6
refbox:
  .zero 64
.align 6
res:
  .zero 512
  TEST_DATA
RVTEST_DATA_END
