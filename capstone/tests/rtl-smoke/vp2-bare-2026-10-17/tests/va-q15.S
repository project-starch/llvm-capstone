# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-q15.S  (Q-15: an inexact CDERIVE child is refused with 29 under the table namespace; the physical profile rounds it, 2026-10-17)
#-----------------------------------------------------------------------------
#
# The setup of va-runv4.S with options 0 and a table of capacity 1024, next 3. First the physical profile's control in M
# (no table bit): CDERIVE of [data+4, data+0x1014) from the data capability succeeds and the child's bounds read back
# rounded outward by the compact format (start data+0, end data+0x1018: E = 0 with the 8-byte granule). Then the
# context derives [data+8, data+0x1018) (representable: an 8-aligned base and an 8-multiple length) and stores 0x77
# through it, and then [data+4, data+0x1014): refused with 29 (the semantics owner's ruling of 2026-10-16, QEMU's
# helper_csderive round trip), a terminal escape at that instruction; the table allocated the first child only.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   the control's child: start minus data 0; end minus data 0x1018;  2 (the legacy-C control);
#   R1 (the refusal): rd 2; +16 2; +24 29; +32 minus ctx_bad 0; data[8] 0x77; root+8 4; the discard 0.

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

# ---------------------------------------------------------------- the table: RECYCLE, capacity 1024, next 3, record 2 the root
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 1024
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

# ---------------------------------------------------------------- the physical profile's control: the child rounds outward
li t0, 4
CINCOFFSET(s6, x26, t0)
li t0, 0x1010
CDERIVE(s4, s6, t0)            # [data+4, data+0x1014) in the physical pool: allowed, rounded at compression
LCC(t1, s4, x3)                # start (the selector is the rs2 FIELD: x3 = 3)
lla t2, data
sub t1, t1, t2
CAPPRINT(t1)                   # 0: rounded down to the 8-byte granule
LCC(t1, s4, x4)                # end (selector 4)
sub t1, t1, t2
CAPPRINT(t1)                   # 0x1018: rounded up

# ---------------------------------------------------------------- page tables as va-runv.S
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

# ---------------------------------------------------------------- the frame
MINT(s9, CAP_TYPE_NONLIN, frame, frame_end, CAP_PERM_RW)
sd s6, 0(s9)
lla t0, tbl
sd t0, 8(s9)
sd x0, 56(s9)                  # options 0
MINT(a2, CAP_TYPE_LIN, ctx, ctx_end, CAP_PERM_RWX)
STC(s9, a2, 64)
li t0, 480
CINCOFFSET(t1, s9, t0)
STC(t1, x26, 0)
li t0, 0x100000
csrw CSUPQUANTUM, t0

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
lla a0, frame
li a1, 0
CSRUNV(a2, a0, a1)             # entry; the refused derive ends it: a2 = 2
mv a3, a2
li a7, 1
ecall                          # R1
li a1, 1
CSRUNV(a2, a0, a1)             # discard: 0
mv a3, a2
li a7, 2
ecall
1: j 1b

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
  li t1, 1
  bne a7, t1, 7f
  CAPPRINT(a3)                 # 2
  ld t2, 16(s9)
  CAPPRINT(t2)                 # 2
  ld t2, 24(s9)
  CAPPRINT(t2)                 # 29
  ld t2, 32(s9)
  lla t3, ctx_bad
  sub t2, t2, t3
  CAPPRINT(t2)                 # 0
  ld t2, 8(x26)
  CAPPRINT(t2)                 # 0x77
  ld t2, 8(s8)
  CAPPRINT(t2)                 # 4: one child allocated
  j 9f
7:
  CAPPRINT(a3)                 # 0
  CAP_PASS(s11)
9:
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 6
ctx:
  li t0, 8
  CINCOFFSET(s6, x26, t0)
  li t0, 0x1010
  CDERIVE(s4, s6, t0)          # [data+8, data+0x1018): representable
  li t1, 0x77
  sd t1, 0(s4)                 # data[8]
  li t0, 4
  CINCOFFSET(s6, x26, t0)
  li t0, 0x1010
ctx_bad:
  CDERIVE(s5, s6, t0)          # [data+4, data+0x1014): 29
  .word 0
  j ctx_bad
ctx_end:

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
data:
  .zero 8192
data_end:
.align 12
tbl:
  .zero 16384
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
