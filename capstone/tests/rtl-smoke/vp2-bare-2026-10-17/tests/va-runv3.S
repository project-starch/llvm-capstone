# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-runv3.S  (vp2 item D, step D3: FP state per context, 2026-10-17)
#-----------------------------------------------------------------------------
#
# The setup of va-runv.S with options 1 and a 4 KiB frame. Before the entry the caller holds f1 = 1.5 and frm = 1. The
# context starts with its FP state zeroed (fpr and fcsr: QEMU's fresh context): it moves f1 to a0, fcsr to a1 and
# ECALLs (kind 3). M reads the context's f1 slot (+664) and fcsr (+912), the caller's f1 (+2056) and fcsr (+2304) beside
# its snapshot, and the caller's live f1 and frm (restored by the escape); then it writes 2.5 into the context's f1
# slot and frm 2 into its fcsr word, a zero reply, and S resumes with action 2: the context reads f1 and fcsr back into
# data[0] and data[8] and ends on an illegal instruction (kind 2). M reads them and the caller's FP state once more.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   2 (the legacy-C control);
#   R1 (the service ECALL): rd 3; +48 (a0: the context's f1) 0; +584 (the a1 view: its fcsr) 0; +664 0; +912 0;
#     +2056 0x3ff8000000000000; +2304 0x20; the caller's f1 0x3ff8000000000000; the caller's frm 1;
#   R2 (the illegal instruction): rd 2; data[0] 0x4004000000000000; data[8] 0x40; the caller's f1
#     0x3ff8000000000000; the caller's frm 1;  the discard: 0.

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
li t0, 0x6000
csrs mstatus, t0               # FS := Dirty: the caller uses FP

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

# ---------------------------------------------------------------- the table: RECYCLE, next 3, record 2 the root
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
li t0, 1
sd t0, 56(s9)                  # options: resumable ECALL
MINT(a2, CAP_TYPE_LIN, ctx, ctx_end, CAP_PERM_RWX)
STC(s9, a2, 64)
li t0, 480
CINCOFFSET(t1, s9, t0)
STC(t1, x26, 0)
li t0, 0x100000
csrw CSUPQUANTUM, t0

# ---------------------------------------------------------------- the caller's FP state
li t0, 0x3ff8000000000000
fmv.d.x f1, t0                 # 1.5
csrwi frm, 1

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
CSRUNV(a2, a0, a1)             # entry; the service ECALL pauses it: a2 = 3
mv a3, a2
li a7, 1
ecall                          # R1; M writes the context's FP words and the reply
li a1, 2
CSRUNV(a2, a0, a1)             # the reply consumed; the illegal instruction ends it: a2 = 2
mv a3, a2
li a7, 2
ecall                          # R2
li a1, 1
CSRUNV(a2, a0, a1)             # discard: 0
mv a3, a2
li a7, 3
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
  bne a7, t1, 6f
  CAPPRINT(a3)                 # 3
  ld t2, 48(s9)
  CAPPRINT(t2)                 # 0: the context's f1 (zeroed at entry)
  ld t2, 584(s9)
  CAPPRINT(t2)                 # 0: its fcsr (the a1 view)
  ld t2, 664(s9)
  CAPPRINT(t2)                 # 0: the f1 slot
  ld t2, 912(s9)
  CAPPRINT(t2)                 # 0: the fcsr word
  li t3, 2048
  CINCOFFSET(t4, s9, t3)       # the frame's second half (offsets above 2047 need a moved capability)
  ld t2, 8(t4)
  CAPPRINT(t2)                 # 0x3ff8000000000000: the caller's f1 beside its snapshot (+2056)
  ld t2, 256(t4)
  CAPPRINT(t2)                 # 0x20: the caller's fcsr (frm 1) at +2304 (fpr0..31 fill +2048..+2303)
  fmv.x.d t2, f1
  CAPPRINT(t2)                 # 0x3ff8000000000000: the caller's live f1
  csrr t2, frm
  CAPPRINT(t2)                 # 1
  li t2, 0x4004000000000000
  sd t2, 664(s9)               # the context's f1 := 2.5
  li t2, 0x40
  sd t2, 912(s9)               # the context's frm := 2
  sd x0, 224(s9)               # the reply
  j 9f
6:
  li t1, 2
  bne a7, t1, 7f
  CAPPRINT(a3)                 # 2
  ld t2, 0(x26)
  CAPPRINT(t2)                 # 0x4004000000000000
  ld t2, 8(x26)
  CAPPRINT(t2)                 # 0x40
  fmv.x.d t2, f1
  CAPPRINT(t2)                 # 0x3ff8000000000000
  csrr t2, frm
  CAPPRINT(t2)                 # 1
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
  fmv.x.d a0, f1
  csrr a1, fcsr
ctx_ecall:
  ecall
  fmv.x.d t1, f1
  sd t1, 0(x26)
  csrr t1, fcsr
  sd t1, 8(x26)
ctx_bad:
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
