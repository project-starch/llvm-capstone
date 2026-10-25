# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-alloc.S  (vp2 item A, step 2: node allocation and retirement driven by the flat table's header, 2026-10-17)
#-----------------------------------------------------------------------------
#
# The step-1 table (va-flat.S) with a free list of two records, 5 -> 4 -> 0 (free_count 2) and next-unused 6, under the
# flag: the domain derives three children from its root (record 2) and the unit must allocate exactly as QEMU's
# table_alloc: ids 5 and 4 from the free list (head first), then 6 by the bump; root+8 becomes 7, root+16 becomes 0
# (head 0, count 0), root+24 becomes 3. Then the domain crevokes the third child (6): the unit clears VALID and sets
# nulls its links (retirement is not a free-list push; collection is software), so record 6's word1 has VALID clear and
# FREE clear and root+16 is unchanged. The domain stores each child's node-bearing reading through s1: the three ids are
# not readable by lcc, so the readings are the children's cursors relative to data (0, 64, 128: the derive placed
# them where the domain asked) and 0x19 for a load through child 6 after its crevoke.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order), flag on:
#   srevroot read back minus the table: 0; status 0; csupcause 2 (since item E, 2026-10-17: the domain's RETURN is refused in a virtual context; 0 before); through s1: 0, 64, 128, 0x19; csupstatus 5 (the escape; 1 before item E);
#   root+0: 0x8000000000000400; root+8: 7; root+16: 0; root+24: 3; record 5 word1 VALID 1, parent 2, depth 1; record 4 word1 VALID 1, parent 2,
#   depth 1; record 6 word0 prev 0 (the retirement nulls the retired record's links, as QEMU's cap_rev_retire);
#   record 6 word1 VALID 0 and FREE 0 ((word1 >> 60) & 7 = 0), parent 2, depth 1;
#   mcause 0.
# Flag OFF (-DVA_FLAG_OFF, the positive control): the same program allocates from the unit's own head in the physical
#   pool (3, 4, 5): through s1 the same 0, 64, 128, 0x19; csupstatus 1; the table as the monitor built it: root+0
#   0x8000000000000400, root+8 6,
# -DVA_CORRUPT (flag on): record 4's word1 is (FREE << 60) | 1, not QEMU's exact FREE record, so the second derive (the
#   pop of 4) is refused with 29 (status 3 -> ILLEGAL_OPERAND_VALUE under the table namespace since D6/Q-20, QEMU's
#   CAP_REV_CORRUPT; 25 before) after the first allocation went through: 0; 0; 29; s1: 0 0 0 0; csupstatus 5; root+0
#   0x8000000000000400; root+8 6; root+16 0x100000004; root+24 1; rec5 1 2 1; rec4 4 0 1; rec6 prev 0; rec6 0 0 0; mcause 0.
# -DVA_EXHAUST (flag on): capacity 263 | RECYCLE, next 6, no free list, records 5 and 4 zero: 257 free identities, one
#   above QEMU's reserve of 256 (D5): the first derive bumps (id 6, next 7), the second finds 256 and is refused with 30
#   (status 2 -> INSUFFICIENT_SYSTEM_RESOURCES, QEMU's node pressure, kind 5 under CSRUNV): 0; 0; 30; s1: 0 0 0 0;
#   csupstatus 5; root+0 0x8000000000000107; root+8 7; root+16 0; root+24 1;
#   rec5 0 0 0; rec4 0 0 0; rec6 prev 2 (live, not retired); rec6 1 2 1; mcause 0.
#   root+16 = (2 << 32) | 5, root+24 0; records 5 and 4 FREE ((word1 >> 60) & 7 = 4, parent 0, depth 0) and record 6's
#   words 0; mcause 0.

# 2026-10-17 (D4/D5): the table's capacity is 1024, not 64: inside a context QEMU's allocator refuses below 257 free
# identities (CAP_REV_TABLE_RESERVE, node pressure 30), and the RTL's does the same since D5; root+0 reads 0x8000000000000400.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define SREVROOT    0x5C1

#ifdef VA_FLAG_OFF
#define VA_CTL_ARM 0
#else
#define VA_CTL_ARM 2
#endif

#define PTE_LEAF_RWX 0xDF
#define PTE_PTR      0x01

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

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

MINT(s5, CAP_TYPE_LIN, savearea, savearea_end, CAP_PERM_RWX)
MINT(s1, CAP_TYPE_LIN, scratch, scratch_end, CAP_PERM_RW)
sd x0, 0(s1)
sd x0, 8(s1)
sd x0, 16(s1)
sd x0, 24(s1)

# ---------------------------------------------------------------- the flat table with a two-record free list
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
#ifdef VA_EXHAUST
ori t0, t0, 263                # capacity 263 | RECYCLE: 257 free identities, one above the reserve
#else
ori t0, t0, 1024               # capacity 1024 | RECYCLE (above the 256-id reserve inside a context)
#endif
sd t0, 0(s8)
li t0, 6
sd t0, 8(s8)                   # next unused 6
#ifdef VA_EXHAUST
sd x0, 16(s8)                  # no free list
#else
li t0, 2
slli t0, t0, 32
ori t0, t0, 5
sd t0, 16(s8)                  # free_head 5, free_count 2
#endif
sd x0, 24(s8)                  # allocations 0
#define REC(IDX, REG) li REG, (IDX)*16; CINCOFFSET(REG, s8, REG)
REC(2, t1)
sd x0, 0(t1)                   # record 2: the root, prev 0 next 0
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)                   # VALID, parent 0, depth 0
#ifdef VA_EXHAUST
REC(5, t1)
sd x0, 0(t1)
sd x0, 8(t1)
REC(4, t1)
sd x0, 0(t1)
sd x0, 8(t1)
#else
REC(5, t1)
li t0, 4
sd t0, 0(t1)                   # record 5: next free = 4
li t0, 4
slli t0, t0, 60
sd t0, 8(t1)                   # FREE << 60
REC(4, t1)
sd x0, 0(t1)                   # record 4: next free = 0 (last)
li t0, 4
slli t0, t0, 60
#ifdef VA_CORRUPT
ori t0, t0, 1                  # not an exact FREE record
#endif
sd t0, 8(t1)
#endif
REC(6, t1)
sd x0, 0(t1)
sd x0, 8(t1)

MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)   # node 2

#ifndef VA_FLAG_OFF
MINT(s7, CAP_TYPE_NONLIN, pt_root, pt_end, CAP_PERM_RW)
lla t0, pt_l0
lla t1, pt_root
sub a7, t0, t1
lla t0, pt_l1
sub s9, t0, t1
li t1, 0x80000000
li t2, 0
1:
srli t3, t1, 12
slli t3, t3, 10
ori t3, t3, PTE_LEAF_RWX
slli t4, t2, 3
add t4, t4, a7
CINCOFFSET(t0, s7, t4)
sd t3, 0(t0)
li t4, 0x1000
add t1, t1, t4
addi t2, t2, 1
li t4, 512
bne t2, t4, 1b
lla t1, pt_l0
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_PTR
CINCOFFSET(t0, s7, s9)
sd t1, 0(t0)
li t2, 1
2:
slli t3, t2, 21
li t4, 0x80000000
add t3, t3, t4
srli t3, t3, 12
slli t3, t3, 10
ori t3, t3, PTE_LEAF_RWX
slli t4, t2, 3
add t4, t4, s9
CINCOFFSET(t0, s7, t4)
sd t3, 0(t0)
addi t2, t2, 1
li t4, 512
bne t2, t4, 2b
lla t1, pt_l1
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_PTR
li t4, 16
CINCOFFSET(t0, s7, t4)
sd t1, 0(t0)
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
lla t0, tbl
csrw SREVROOT, t0
csrr t1, SREVROOT
sub t1, t1, t0
CAPPRINT(t1)                   # 0
#endif

li t0, 0x100000
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, VA_CTL_ARM

CAPCREATE(a2); li a0, CAP_TYPE_LIN; lla a3, stub0; lla a4, stub0_end; li a5, CAP_PERM_RWX
CAPTYPE(a2, a0); CAPBOUND(a2, a3, a4); CAPPERM(a2, a5)
CAPCREATE(a1); li a0, CAP_TYPE_LIN; lla a3, seal0; lla a4, seal0_end; li a5, CAP_PERM_RWX
CAPTYPE(a1, a0); CAPBOUND(a1, a3, a4); CAPPERM(a1, a5)
STC(a1, a2, 0); li a0, 0x0000000A00040000; sd a0, 48(a1); sd x0, 56(a1); sd x0, 64(a1); sd x0, 72(a1); sd x0, 80(a1)
SEAL(a3, a1); csrw mcause, x0; CSSUPERVISE(t3, a3, s5); addi s2, t3, 0; CAPPRINT(s2); CALL(a3, a3)
csrr t0, CSUPSTATUS
csrr t2, CSUPCAUSE
CAPPRINT(t2)                   # 0
ld t1, 0(s1)
CAPPRINT(t1)                   # 0
ld t1, 8(s1)
CAPPRINT(t1)                   # 64
ld t1, 16(s1)
CAPPRINT(t1)                   # 128
ld t1, 24(s1)
CAPPRINT(t1)                   # 0x19
CAPPRINT(t0)                   # 1

# ---------------------------------------------------------------- the header and the records after the domain
ld t2, 0(s8)
CAPPRINT(t2)                   # 0x8000000000000400 (both: capacity | RECYCLE survives the slot-0 write; 0x...107 with VA_EXHAUST)
ld t2, 8(s8)
CAPPRINT(t2)                   # 7 (flag on) / 6 (off)
ld t2, 16(s8)
CAPPRINT(t2)                   # 0 (on) / 0x200000005 (off)
ld t2, 24(s8)
CAPPRINT(t2)                   # 3 (on) / 0 (off)
#define REC_READ(IDX) REC(IDX, t1); ld t2, 8(t1); srli t3, t2, 60; andi t3, t3, 7; CAPPRINT(t3); srli t3, t2, 30; li t4, 0xffff; and t3, t3, t4; CAPPRINT(t3); li t4, 0x1ffff; and t3, t2, t4; CAPPRINT(t3)
REC_READ(5)                    # 1 2 1 (on) / 4 0 0 (off)
REC_READ(4)                    # 1 2 1 (on) / 4 0 0 (off)
REC(6, t1)
ld t2, 0(t1)
li t4, 0xffff
and t3, t2, t4
CAPPRINT(t3)                   # 0 (both): record 6's prev, nulled by the retirement (QEMU cap_rev_retire) / untouched
REC_READ(6)                    # 0 2 1 (on: retired, not free) / 0 0 0 (off)
csrr t0, mcause
CAPPRINT(t0)                   # 0
CAP_PASS(s11)

trap_handler:
  csrr t0, mcause
  CAPPRINT(t0)
  CAP_PASS(s11)

.align 6
stub0:
  MOVC(s3, x1)
  li t0, 64
  CDERIVE(s4, x26, t0)         # child A at data+0
  LCC(t1, s4, x2)              # its cursor
  lla t2, data
  sub t1, t1, t2
  sd t1, 0(s1)                 # 0
  CINCOFFSET(s6, x26, t0)      # data+64
  CDERIVE(s4, s6, t0)          # child B at data+64 (derived from the root's cursor moved: still the root's node)
  LCC(t1, s4, x2)
  sub t1, t1, t2
  sd t1, 8(s1)                 # 64
  li t0, 128
  CINCOFFSET(s6, x26, t0)
  li t0, 64
  CDERIVE(s4, s6, t0)          # child C at data+128
  LCC(t1, s4, x2)
  sub t1, t1, t2
  sd t1, 16(s1)                # 128
  CREVOKE(x26, s4)             # child C retired
  li t0, 0x19
  sd t0, 24(s1)
  MOVC(ra, s3)
  la a1, stub0
  RETURN(ra, a1, x0)
stub0_end:

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
scratch:
  .zero 64
scratch_end:
.align 12
data:
  .zero 4096
data_end:
.align 12
tbl:
  .zero 4096
tbl_end:
.align 12
pt_root:
  .zero 4096
.align 12
pt_l1:
  .zero 4096
.align 12
pt_l0:
  .zero 4096
pt_end:
.align 6
seal0:
  .zero 2048
seal0_end:
  TEST_DATA
RVTEST_DATA_END
