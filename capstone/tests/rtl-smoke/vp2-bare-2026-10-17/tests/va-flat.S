# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-flat.S  (vp2 item A, step 1: the rev-node unit reads and writes the virtual context's FLAT lifetime table, 2026-10-17)
#-----------------------------------------------------------------------------
#
# The monitor builds, in its own data, a flat lifetime table in QEMU's layout (cap_rev_table_abi.h): slot 0 = capacity
# with RECYCLE (bit 63), slot 0's second word next-unused = 3, slot 1 = free list (head 0, count 0) and allocations 0,
# and record 2 = the context's root node: word 0 {next 0, prev 0}, word 1 {VALID, parent 0, depth 0}. It mints the
# domain's root capability over `data`, whose node IS 2 (every CAPCREATE'd capability carries node 2), points srevroot at the table, sets
# csupctl.virt, builds the identity page tables of vb-trans.S and arms a supervised domain. Under the flag every node
# access of the domain's lifetime instructions goes to the table (ex_stage.sv, step 1): the unit is unchanged, so new
# ids still come from its own allocator (its generation rides in the record's spare parent bits, a scaffold step 2 removes).
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order), flag on:
#   srevroot read back minus the table's address: 0 (diagnostic, flag-on build only); status 0 (armed); csupcause 2 (since item E, 2026-10-17: the domain's RETURN is refused in a virtual context and it escapes with cause 2; 0 before);
#   the domain, through s1: the child's perms 0xe (cderive from the root: the query read record 2 as
#     VALID), 0x77 (stored and loaded through the child), 0x19 (a load through the child after crevoke: the kill
#     walked record 2's junior run in the table); csupstatus 5 (the refused RETURN's escape; 1 before item E)
#   the monitor, reading the table bare through its own capability after the RETURN:
#     record 2's word 1 has VALID set and parent 0 (the root survived: (word1 >> 60) & 7 = 1, (word1 >> 30) & 0xffff = 0)
#     record 3 (the child the unit allocated at its own next index, 3 = the unit's head after the monitor's mints):
#       word 0: prev = 2 ((word0 & 0xffff) = 2); word 1: VALID clear after the crevoke ((word1 >> 60) & 1 = 0; the unit
#       set FREE there when it pushed the node to ITS free list, the step-1 divergence), parent's low 16 bits = 2, depth = 1
#   Read 2026-10-17 exactly so (run s1 on virtual-profile-2): 0 0 0 0xe 0x77 0x19 1 1 0 2 0 2 1 0; the flag-off control
#   0 0 0xe 0x77 0x19 1 1 0 0 0 0 0 0. The channel trace showed the derive writing record 3 {prev 2, VALID, parent 2,
#   depth 1} and record 2's next := 3, the kill writing record 3 {FREE}, and no write to slot 0 or 1.
#   mcause 0
# Flag OFF (-DVA_FLAG_OFF, the positive control): x26's node 2 is read from the PHYSICAL pool, where index 2 is the live
#   genesis node, so the domain's readings are the same (0xe, 0x77, 0x19) and the table is UNTOUCHED (record 2 as the
#   monitor wrote it: 1, 0; record 3's words 0, 0, 0, 0): the child lived in the pool; csupstatus 1; mcause 0.
#   (A first cut set the node with CAPNODE from an integer register, which copies a capability's node and so gave 0: the
#   query then read the table's header as a dead node; the channel print showed idx=0.)
# The step-1 scaffold's divergences from QEMU, recorded here and closed by step 2: the child's id comes from the unit's
# head (3 here by construction, not from root+8), root+8 is not advanced, and the crevoked child is pushed to the unit's
# free list, not the table's.

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

# ---------------------------------------------------------------- the flat table (bare, through s8)
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 1024               # capacity 1024 slots | RECYCLE (above the 256-id reserve inside a context)
sd t0, 0(s8)
li t0, 3
sd t0, 8(s8)                   # next unused = 3 (ids 0 and 1 reserved, 2 is the root)
sd x0, 16(s8)                  # free list: head 0, count 0
sd x0, 24(s8)                  # allocations 0
li t0, 32
CINCOFFSET(t1, s8, t0)
sd x0, 0(t1)                   # record 2 word 0: prev 0, next 0
li t0, 1
slli t0, t0, 60                # VALID, parent 0, depth 0
sd t0, 8(t1)

# the domain's root: NONLIN RW + MANAGE over `data`, its node = record 2
MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)   # a CAPCREATE'd capability carries node 2 (capstone_flu_unit.anvil CAPCREATE)

#ifndef VA_FLAG_OFF
# ---------------------------------------------------------------- page tables as vb-trans.S (identity, USER), PMP, satp
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
CAPPRINT(t1)                   # 0: the base read back equals the table's address (diagnostic)
#endif

li t0, 0x100000
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, VA_CTL_ARM

# ---------------------------------------------------------------- the arm
CAPCREATE(a2); li a0, CAP_TYPE_LIN; lla a3, stub0; lla a4, stub0_end; li a5, CAP_PERM_RWX
CAPTYPE(a2, a0); CAPBOUND(a2, a3, a4); CAPPERM(a2, a5)
CAPCREATE(a1); li a0, CAP_TYPE_LIN; lla a3, seal0; lla a4, seal0_end; li a5, CAP_PERM_RWX
CAPTYPE(a1, a0); CAPBOUND(a1, a3, a4); CAPPERM(a1, a5)
STC(a1, a2, 0); li a0, 0x0000000A00040000; sd a0, 48(a1); sd x0, 56(a1); sd x0, 64(a1); sd x0, 72(a1); sd x0, 80(a1)
SEAL(a3, a1); csrw mcause, x0; CSSUPERVISE(t3, a3, s5); addi s2, t3, 0; CAPPRINT(s2); CALL(a3, a3)
csrr t0, CSUPSTATUS
csrr t2, CSUPCAUSE
CAPPRINT(t2)                   # 0 on a return (diagnostic: the cause of an escape)
ld t1, 0(s1)
CAPPRINT(t1)                   # 0xe
ld t1, 8(s1)
CAPPRINT(t1)                   # 0x77
ld t1, 16(s1)
CAPPRINT(t1)                   # 0x19
CAPPRINT(t0)                   # 1

# ---------------------------------------------------------------- the table after the domain
li t0, 32
CINCOFFSET(t1, s8, t0)
ld t2, 8(t1)
srli t3, t2, 60
andi t3, t3, 7
CAPPRINT(t3)                   # 1: record 2 VALID
srli t3, t2, 30
li t4, 0xffff
and t3, t3, t4
CAPPRINT(t3)                   # 0: record 2's parent (low 16)
li t0, 48
CINCOFFSET(t1, s8, t0)
ld t2, 0(t1)
li t4, 0xffff
and t3, t2, t4
CAPPRINT(t3)                   # 2: record 3's prev (flag on) / 0 (flag off)
ld t2, 8(t1)
srli t3, t2, 60
andi t3, t3, 1
CAPPRINT(t3)                   # 0: record 3 VALID clear after the crevoke (both)
srli t3, t2, 30
and t3, t3, t4
CAPPRINT(t3)                   # 2: record 3's parent (flag on) / 0 (flag off)
li t4, 0x1ffff
and t3, t2, t4
CAPPRINT(t3)                   # 1: record 3's depth (flag on) / 0 (flag off)
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
  CDERIVE(s4, x26, t0)         # the child: the root's liveness is record 2 in the table
  LCC(t1, s4, x5)              # selector 5 (the register NUMBER selects the field): the child's perms, 0xe
  sd t1, 0(s1)
  li t1, 0x77
  sd t1, 0(s4)
  ld t2, 0(s4)
  sd t2, 8(s1)
  CREVOKE(x26, s4)             # the kill walks record 2's junior run in the table
  li t2, 0x19
  sd t2, 16(s1)
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
