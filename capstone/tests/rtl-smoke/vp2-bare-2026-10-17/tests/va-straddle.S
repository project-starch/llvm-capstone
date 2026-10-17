# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-straddle.S  (vp2 item A, step 2 audit: a rev-node handler that straddles the flag's fall, 2026-10-17)
#-----------------------------------------------------------------------------
#
# vctx_eff drops the cycle the domain switcher goes busy, and lifetime instructions issue speculatively, so a CDERIVE
# right behind a load that page-faults can have its handler running in the rev-node unit when the escape flips the
# flag: without a hold, the handler's remaining accesses land in the PHYSICAL pool at the virtual indices, and the
# allocator's header writes land on physical slots 0 and 1 (node 1 is a live genesis node). The domain does exactly
# that: a load through s4 (the `nomap` page, unmapped) then a CDERIVE from its root. After the escape the monitor reads
# the table's next-unused and the physical pool's node 1 (word 0 bit 1 is VALID in the physical layout) through a
# capability over CAP_REVNODE_MEM_BASE.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order): status 0; csupstatus 5; csupcause 13
#   (load page fault); root+8: 7 if the derive's handler ran to completion in the table, 6 if it never reached the
#   unit; physical node 1's word 0 & 2: 2 (intact); physical node 2's next: 0 (a derive that ran after the flag
#   dropped would have allocated a physical child of the monitor's node 2 and linked it there). Before the hold fix a
#   straddle that reaches the unit reads node 1 clobbered (0) and root+8 6; whether the straddle is reachable in this
#   simulation is itself the reading, so the fixed tree must read 2 and 0 in every case and the unfixed tree's reading
#   is recorded as found (2026-10-17, unfixed: 0 5 0xd 6 2 0: the derive ran in neither namespace, consistent with the
#   dyn unit abandoning a flushed handler before its request, the `loop ep.flush` idiom). mcause 0.

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
MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)   # node 2
MINT(s4, CAP_TYPE_NONLIN, nomap, nomap_end, CAP_PERM_RW)

# the physical pool, bare, for the readout
CAPCREATE(a6); li a0, CAP_TYPE_NONLIN; li a3, 0xBFF00000; li a4, 0xBFF00040; li a5, CAP_PERM_RW
CAPTYPE(a6, a0); CAPBOUND(a6, a3, a4); CAPPERM(a6, a5)

# ---------------------------------------------------------------- the table: record 2 live, next-unused 6, no free list
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 1024               # capacity 1024 | RECYCLE (above the 256-id reserve inside a context)
sd t0, 0(s8)
li t0, 6
sd t0, 8(s8)
sd x0, 16(s8)
sd x0, 24(s8)
li t0, 32
CINCOFFSET(t1, s8, t0)
sd x0, 0(t1)
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)                   # record 2: VALID, parent 0, depth 0

# ---------------------------------------------------------------- page tables as va-alloc.S, then `nomap` unmapped
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
# L0[nomap] -> invalid
lla t0, nomap
li t1, 0x80000000
sub t0, t0, t1
srli t0, t0, 12
slli t0, t0, 3
add t0, t0, a7
CINCOFFSET(t2, s7, t0)
sd x0, 0(t2)
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

li t0, 0x100000
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, 2

CAPCREATE(a2); li a0, CAP_TYPE_LIN; lla a3, stub0; lla a4, stub0_end; li a5, CAP_PERM_RWX
CAPTYPE(a2, a0); CAPBOUND(a2, a3, a4); CAPPERM(a2, a5)
CAPCREATE(a1); li a0, CAP_TYPE_LIN; lla a3, seal0; lla a4, seal0_end; li a5, CAP_PERM_RWX
CAPTYPE(a1, a0); CAPBOUND(a1, a3, a4); CAPPERM(a1, a5)
STC(a1, a2, 0); li a0, 0x0000000A00040000; sd a0, 48(a1); sd x0, 56(a1); sd x0, 64(a1); sd x0, 72(a1); sd x0, 80(a1)
SEAL(a3, a1); csrw mcause, x0; CSSUPERVISE(t3, a3, s5); addi s2, t3, 0; CAPPRINT(s2); CALL(a3, a3)
csrr t0, CSUPSTATUS
CAPPRINT(t0)                   # 5
csrr t2, CSUPCAUSE
CAPPRINT(t2)                   # 13
ld t2, 8(s8)
CAPPRINT(t2)                   # 7 (the derive completed in the table) or 6 (it never reached the unit)
ld t2, 16(a6)
andi t2, t2, 2
CAPPRINT(t2)                   # 2: physical node 1 still VALID
ld t2, 32(a6)
srli t2, t2, 2
slli t2, t2, 34
srli t2, t2, 34
CAPPRINT(t2)                   # 0: physical node 2's next (word 0 [31:2]): no derive ran in the PHYSICAL namespace
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
  ld t1, 0(s4)                 # page fault: the escape
  CDERIVE(s6, x26, t0)         # issued behind it
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
nomap:
  .zero 4096
nomap_end:
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
