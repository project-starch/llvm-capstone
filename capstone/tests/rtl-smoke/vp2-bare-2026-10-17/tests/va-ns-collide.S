# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-ns-collide.S  (vp2 item A, step 2 audit: the revocation trackers across a namespace change, 2026-10-17)
#-----------------------------------------------------------------------------
#
# Under the flag ids carry no generation, so a virtual id (0, i) and the physical pool's (0, i) are the same 30-bit tag in
# the LSU's revocation cache and the CPMP tracker, and neither has a context term. The arm makes the collision: the
# monitor (flag off) derives three physical children of its root (ids 3, 4, 5), loads through child 5 so the cache holds
# (0, 5) LIVE, then builds a table whose record 5 is RETIRED (VALID clear) and gives the domain a root capability on node
# 5 (CAPNODE from the physical child). Under the flag the domain loads through it.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   before the fix (a tracker that survives the namespace change): status 0; csupstatus 1 (the domain returned);
#     csupcause 0; s1[0] 0x77 (the load was ALLOWED on the stale physical entry); mcause 0.
#   with the trackers flushed on the namespace change: status 0; csupstatus 5; csupcause 25 (the load's liveness was
#     re-evidenced from the table: record 5 is dead); s1[0] 0; mcause 0.
#   The physical control (-DVA_FLAG_OFF): the same program with the flag off: the domain's x26 is the live physical
#     child 5, so csupstatus 1, csupcause 0, s1[0] 0x77 on both trees.

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

# ---------------------------------------------------------------- the physical children (flag off): ids 3, 4, 5
MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)   # node 2
li t0, 0x77
sd t0, 0(x26)
li t0, 64
CDERIVE(s6, x26, t0)           # physical 3
CDERIVE(s6, x26, t0)           # physical 4
CDERIVE(s6, x26, t0)           # physical 5 (the last derive's child stays in s6)
ld t1, 0(s6)                   # through child 5: the revocation cache now holds (0, 5) LIVE
CAPPRINT(t1)                   # 0x77

# ---------------------------------------------------------------- the table: record 2 live, record 5 RETIRED
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 1024
sd t0, 0(s8)                   # capacity 1024 | RECYCLE (above the 256-id reserve inside a context)
li t0, 6
sd t0, 8(s8)                   # next unused 6
sd x0, 16(s8)                  # no free list
sd x0, 24(s8)
#define REC(IDX, REG) li REG, (IDX)*16; CINCOFFSET(REG, s8, REG)
REC(2, t1)
sd x0, 0(t1)
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)                   # record 2: VALID, parent 0, depth 0
REC(5, t1)
li t0, 2
sd t0, 0(t1)                   # record 5: prev 2
li t0, 2
slli t0, t0, 30
ori t0, t0, 1
sd t0, 8(t1)                   # record 5: VALID clear, parent 2, depth 1 (retired)

#ifndef VA_FLAG_OFF
CAPNODE(x26, s6)               # the domain's root carries node 5: dead in the table, live in the physical pool
#else
MOVC(x26, s6)                  # the control: the live physical child itself
#endif

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
CAPPRINT(t0)                   # 1 (stale ALLOW) / 5 (flushed)
csrr t2, CSUPCAUSE
CAPPRINT(t2)                   # 0 / 25
ld t1, 0(s1)
CAPPRINT(t1)                   # 0x77 / 0
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
  ld t1, 0(x26)                # through node 5
  sd t1, 0(s1)
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
