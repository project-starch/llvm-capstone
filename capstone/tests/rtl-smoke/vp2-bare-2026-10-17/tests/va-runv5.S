# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-runv5.S  (vp2 item E: the virtual context's instruction policy and #25's untagged arithmetic, 2026-10-17)
#-----------------------------------------------------------------------------
#
# The setup of va-runv.S with options 0. Six entries, one context each (the PCC re-minted before each): the context
# executes an instruction QEMU's virtual-C filter refuses and the RTL now refuses under the flag, each a terminal
# escape with cause 2 (illegal instruction): (1) seal, (2) return, (3) ccsrrw of cpmp0, (4) sfence.vma, (5) a read of
# CSR 0x803 (offsetmmu). (6) #25: an untagged cincoffset, cincoffsetimm and scc give integer results under the flag
# (5 + 7 = 12, 12 + 3 = 15, scc 9 -> 9), stored into data[0..16], then an illegal instruction ends it (cause 2).
# Between entries S discards the slot.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   2 (the legacy-C control);
#   entries 1..5: rd 2; +16 2; +24 2; +32 minus the context's address 0; the discard 0;
#   entry 6: rd 2; +24 2; data[0] 12; data[8] 15; data[16] 9; the discard 0.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSRUNV(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x24, rd, rs1, rs2

#define CSUPQUANTUM 0x7C3

#define PTE_LEAF_RWX  0xDF
#define PTE_LEAF_SRWX 0xCF
#define PTE_PTR       0x01

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

MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 64
sd t0, 0(s8)
li t0, 3
sd t0, 8(s8)
sd x0, 16(s8)
sd x0, 24(s8)
li t0, 32
CINCOFFSET(t1, s8, t0)
sd x0, 0(t1)
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)

MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)
sd x0, 0(x26)
sd x0, 8(x26)
sd x0, 16(x26)

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
lla t0, s_code
li t1, 0x80000000
sub t0, t0, t1
srli t0, t0, 12
slli t0, t0, 3
add t0, t0, a7
CINCOFFSET(t2, s7, t0)
lla t1, s_code
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_LEAF_SRWX
sd t1, 0(t2)
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
MINT(a1, CAP_TYPE_NONLIN, _start, pt_end, CAP_PERM_RWX)
CCSRRW(x0, CCSR_CPMP(0), a1)
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
mv s6, t0

MINT(s9, CAP_TYPE_NONLIN, frame, frame_end, CAP_PERM_RW)
sd s6, 0(s9)
lla t0, tbl
sd t0, 8(s9)
sd x0, 56(s9)
li t0, 480
CINCOFFSET(t1, s9, t0)
STC(t1, x26, 0)
li t0, 0x100000
csrw CSUPQUANTUM, t0

# the first context's PCC
#define PCC(LO, HI) CAPCREATE(s3); li t0, CAP_TYPE_LIN; CAPTYPE(s3, t0); lla t0, LO; lla t1, HI; CAPBOUND(s3, t0, t1); li t0, CAP_PERM_RWX; CAPPERM(s3, t0); STC(s9, s3, 64)
PCC(ctx1, ctx1_end)

lla a0, frame
li a1, 0
CSRUNV(a2, a0, a1)             # legacy C: illegal (2)

li t0, 0x1800
csrc mstatus, t0
li t0, 0x0800
csrs mstatus, t0
lla t0, s_code
csrw mepc, t0
mret

.align 12
s_code:
li a7, 1
3:
lla a0, frame
li a1, 0
CSRUNV(a2, a0, a1)             # entry; the refused instruction ends it: a2 = 2
mv a3, a2
li a1, 1
CSRUNV(a4, a0, a1)             # discard: 0
ecall                          # the readings; M mints the next context's PCC
addi a7, a7, 1
j 3b

.align 12
trap_handler:
  csrr t0, mcause
  li t1, 9
  beq t0, t1, 5f
  CAPPRINT(t0)                 # 2: the control
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
5:
  CAPPRINT(a3)                 # 2
  ld t2, 16(s9)
  CAPPRINT(t2)                 # 2
  ld t2, 24(s9)
  CAPPRINT(t2)                 # 2
  li t1, 6
  beq a7, t1, 6f
  ld t2, 32(s9)
  lla t3, ctx1
  li t1, 1
  beq a7, t1, 4f
  lla t3, ctx2
  li t1, 2
  beq a7, t1, 4f
  lla t3, ctx3
  li t1, 3
  beq a7, t1, 4f
  lla t3, ctx4
  li t1, 4
  beq a7, t1, 4f
  lla t3, ctx5
4:
  sub t2, t2, t3
  CAPPRINT(t2)                 # 0: the refused instruction is the context's first
  CAPPRINT(a4)                 # 0: the discard
  li t1, 1
  bne a7, t1, 11f
  PCC(ctx2, ctx2_end)
  j 9f
11:
  li t1, 2
  bne a7, t1, 12f
  PCC(ctx3, ctx3_end)
  j 9f
12:
  li t1, 3
  bne a7, t1, 13f
  PCC(ctx4, ctx4_end)
  j 9f
13:
  li t1, 4
  bne a7, t1, 14f
  PCC(ctx5, ctx5_end)
  j 9f
14:
  PCC(ctx6, ctx6_end)
  j 9f
6:
  ld t2, 0(x26)
  CAPPRINT(t2)                 # 12
  ld t2, 8(x26)
  CAPPRINT(t2)                 # 15
  ld t2, 16(x26)
  CAPPRINT(t2)                 # 9
  CAPPRINT(a4)                 # 0
  CAP_PASS(s11)
9:
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 6
ctx1:
  SEAL(t1, x26)                # refused: 2
  j ctx1
ctx1_end:
.align 6
ctx2:
  RETURN(ra, x26, x0)          # refused: 2
  j ctx2
ctx2_end:
.align 6
ctx3:
  CCSRRW(t1, CCSR_CPMP(0), x0) # refused: 2
  j ctx3
ctx3_end:
.align 6
ctx4:
  sfence.vma                   # refused: 2
  j ctx4
ctx4_end:
.align 6
ctx5:
  csrr t1, 0x803               # refused: 2
  j ctx5
ctx5_end:
.align 6
ctx6:
  li t0, 5
  li t1, 7
  CINCOFFSET(t2, t0, t1)       # 12: an untagged rs1 adds as an integer
  sd t2, 0(x26)
  CINCOFFSETIMM(t2, t2, 3)     # 15
  sd t2, 8(x26)
  li t1, 9
  SCC(t2, t0, t1)              # 9
  sd t2, 16(x26)
  .word 0                      # 2
  j ctx6
ctx6_end:

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
data:
  .zero 4096
data_end:
.align 12
tbl:
  .zero 4096
tbl_end:
.align 12
frame:
  .zero 4096
frame_end:
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
  TEST_DATA
RVTEST_DATA_END
