# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-cptag.S -- B5b-2 step 6, part 1: CPTAG's ordering and the page bitmap
#-----------------------------------------------------------------------------
#
# The reference (capstone-qemu 9289114e06 trans_cptag, op_helper.c :3573-3585, cap_mem_map.c :504-522): CPTAG rd, rs1,
# mode (custom-0 funct3 6, I format; mode = the immediate, 0 or 1) answers whether the 4 KiB physical page at rs1 may
# hold a tag. rs1 is physical (no translation, no PMP); a page not aligned or not wholly inside the tagged region gives
# a load access fault (5) with tval = rs1; any other immediate, U, protected U or a context gives 2; rd = x0 writes
# nothing. Part 1 answers with the adapter's page bitmap (sticky since reset; the acknowledge's exact OR is part 2),
# ordered behind every older tag write (the semantics owner's ruling; the hazard: a page's bit is set only at the end of
# its TAG_FILL, after the data write's B).
# Silicon's tagged region is [0x8000_0000, 0xBC2D_2D2D) (MEMORY_TOP = the tag base): the last whole page inside is
# 0xBC2D_1000. A QEMU machine's region is its main RAM (-m), a machine-configuration difference.
# b5b2-sref.S's setup: x26/s10 = {0,3} live over data. a1 = pg_never (never written), a2 = pg_tag (M stc of x26 makes it
# tagged), a3 = pg_cache. Traps go to M; the handler records mcause in x9/s1 and mtval in x19/s3, counts in x28/t3,
# resumes at mepc + 4. Readings are stored at x25/s9 (U holds its own in a0) and printed by M at the end, after their
# count. t1 is set to 0x5a5a before each CPTAG that has a result reading.
# Readings:
#   1 cause / 2 result of CPTAG pg_never
#   3 cause of CPTAG pg_tag right after the page's first tagged store / 4 add t2, t1, x0 right after it (a reader of rd
#     waits for the answer, never rs1)
#   5 the same CPTAG, its result read after a 20-iteration delay
#   6 cause / 7 result of CPTAG pg_tag, mode 1 (acknowledge; the tag is still there)
#   8 cause of CPTAG with rd = x0
#   9 / 10 cause, mtval == rs1 for pg_tag + 8 (unaligned)      11 / 12 for 0x7FFF_F000 (below the region)
#   13 / 14 for 0xBC2D_2000 (straddles the region's top)       15 / 16 for 0xBC2D_2D2D (the top itself)
#   17 cause / 18 result for 0xBC2D_1000 (the last page inside; never tagged)
#   19 cause of CPTAG with immediate 2
#   20 a cached word at pg_cache reads 0x1234 after CPTAG pg_cache   21 (its load's cycles after) - (before) <= 2: the
#      cache line at rs1 is neither invalidated nor rewritten (a miss would cost a memory round trip)
#   22 cause of CPTAG in protected U (srevroot = tbl)
#   23 the trap count
# PRE-REGISTERED (2026-11-24, written before any run):
#   step 6 part 1:  0 0 | 0 1 | 1 | 0 1 | 0 | 5 1 5 1 5 1 5 1 | 0 0 | 2 | 1 1 | 2 | 6
#     and in the simulation log: at least one "CPTAG-WAIT" line (the hazard window exercised: the CPTAG reached the
#     adapter while an older tag write was in flight), and exactly 7 "CPTAG-ANSWER" lines (readings 1, 3, 5, 6, 8, 17, 20's
#     CPTAG), bit 0 for pg_never, 0xBC2D_1000 and pg_cache, 1 for pg_tag.
#   base (b5b2-step5 4457a77a7: funct3 6 is illegal, every CPTAG traps 2 and is skipped):
#     2 0x5a5a | 2 0x5a5a | 0x5a5a | 2 0x5a5a | 2 | 2 0 2 0 2 0 2 0 | 2 0x5a5a | 2 | 1 1 | 2 | 13
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
#define EQ(R, V)  sub a5, R, V; seqz a5, a5; REC
#define CAUSE     mv a5, x9; REC
#define TO_U(LBL, BACK) li t0, 0x1800; csrc mstatus, t0; lla t0, LBL; csrw mepc, t0; lla x30, BACK; mret

lla a1, pg_never
lla a2, pg_tag
lla a3, pg_cache
csrw SREVROOT, x0

# 1, 2: a page never tagged
li t1, 0x5a5a; CLR; CPTAG(t1, a1, 0); CAUSE; mv a5, t1; REC
# 3, 4: the hazard -- the page's first tagged store, then CPTAG at once and a reader right behind it
STC(a2, x26, 0)
li t1, 0x5a5a; CLR; CPTAG(t1, a2, 0); add t2, t1, x0
CAUSE; mv a5, t2; REC
# 5: the result read after a delay
li t1, 0x5a5a; CPTAG(t1, a2, 0)
li a4, 20
1: addi a4, a4, -1
bnez a4, 1b
add t2, t1, x0; mv a5, t2; REC
# 6, 7: acknowledge, the tag still there
li t1, 0x5a5a; CLR; CPTAG(t1, a2, 1); CAUSE; mv a5, t1; REC
# 8: rd = x0
CLR; CPTAG(x0, a2, 0); CAUSE
# 9-16: faults, tval = rs1
addi a4, a2, 8
CLR; CPTAG(t1, a4, 0); CAUSE; EQ(x19, a4)
li a4, 0x7ffff000
CLR; CPTAG(t1, a4, 0); CAUSE; EQ(x19, a4)
li a4, 0xbc2d2000
CLR; CPTAG(t1, a4, 0); CAUSE; EQ(x19, a4)
li a4, 0xbc2d2d2d
CLR; CPTAG(t1, a4, 0); CAUSE; EQ(x19, a4)
# 17, 18: the last page inside
li a4, 0xbc2d1000
li t1, 0x5a5a; CLR; CPTAG(t1, a4, 0); CAUSE; mv a5, t1; REC
# 19: immediate 2
CLR; .insn i 0x0b, 6, t1, a2, 2; CAUSE
# 20, 21: the line at rs1 is not touched
li a4, 0x1234
sd a4, 0(a3)
ld x22, 0(a3)
rdcycle a6; ld x22, 0(a3); rdcycle a7; sub x31, a7, a6
CPTAG(t1, a3, 0)
rdcycle a6; ld x23, 0(a3); rdcycle a7; sub a7, a7, a6
EQ(x23, a4)
sub a6, a7, x31; slti a5, a6, 3; REC
# 22: protected U
lla t0, tbl
csrw SREVROOT, t0
li a0, 0
TO_U(u_cptag, m_after_u)
m_after_u:
csrw SREVROOT, x0
mv a5, a0; REC
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
u_cptag:
  # U records in a0 only: a store through x25, an integer base, would itself raise 24 in protected U
  CLR; CPTAG(t1, a2, 0); mv a0, x9
  ecall

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
pg_never:
  .zero 4096
pg_tag:
  .zero 4096
pg_cache:
  .zero 4096
.align 6
refbox:
  .zero 64
.align 6
res:
  .zero 512
  TEST_DATA
RVTEST_DATA_END
