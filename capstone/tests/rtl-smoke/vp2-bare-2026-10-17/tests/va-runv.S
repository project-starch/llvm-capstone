# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-runv.S  (vp2 item D, step D1: CSRUNV entry from scalar S, the escape into the frame, action 1, 2026-10-17)
#-----------------------------------------------------------------------------
#
# The monitor (M, capability mode) builds a RECYCLE lifetime table (record 2 the root), identity page tables with USER
# leaves (the S-mode code's page without U, since S cannot fetch from a user page), CPMP0 over the whole image (S-mode
# code is confined by the capability PMP in capability mode), and a 4 KiB frame (the caller's FP state at +2048): +0 satp, +8 the
# table, +56 options 0, +64 the PCC (a tagged LINEAR RWX capability over `ctx`), +480 the x26 slot (a tagged NONLIN RW
# over `data`, node 2). It writes 0x77 into data[0], then mret's into scalar S code, which runs CSRUNV rd=a2, rs1=the
# frame's physical address, rs2=0. The context (C mode, translated at U) loads data[0] through x26, stores it at
# data[8], sets a0 = 0x55 and x11 = the loaded word, and executes ecall: options bit 0 is clear, so the escape is
# terminal (kind 2, cause 11). S resumes at CSRUNV+4 with a2 = 2 and ecalls into M, which reads the frame through the
# caller's restored s9, then S runs action 1 (discard, a2 = 0). First, in M after CAPENTER (legacy C), CSRUNV is
# illegal: the control reads mcause 2 from the trap handler.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   mcause 2 (the legacy-C control); then after the first escape: a2 (the kind) 2; the caller's s2 0x1234 (restored
#   from the frame's second kilobyte); the frame's +16 kind 2, +24 cause 11, +32 pc minus ctx_ecall 0, +40 addr 0,
#   +48 raw a0 0x55; the PCC slot's cursor at +64 minus ctx_ecall 0; the x11 slot at +240 0x77; data[8] 0x77;
#   csupstatus 5 and csupcause 11 (the escape's CSR record, written beside the frame); then after action 1: a2 0;
#   mcause (the handler's own, the second ecall) 9.
# -DVA_QUANTUM: csupquantum 3000 and the context loops: the escape is kind 1 (the quantum): a2 1; s2 0x1234; +16 1;
#   +24 0 (QEMU's quantum event; csupcause keeps the mark); +32 minus ctx_loop 0; +40 0; +48 0 (a0 untouched); PCC cursor minus
#   ctx_loop 0; +240 0; data[8] 0 (never reached); csupstatus 3; csupcause 0x8000000000000010; then action 1: 0; 9.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSRUNV(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x24, rd, rs1, rs2

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define SREVROOT    0x5C1

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

# ---------------------------------------------------------------- data, and the context's capability over it (node 2)
MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)
li t0, 0x77
sd t0, 0(x26)
sd x0, 8(x26)

# ---------------------------------------------------------------- page tables as va-alloc.S; the S-mode page without U
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
# L0[s_code] -> no U
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
# the capability PMP: S-mode code is confined by CPMP in capability mode; grant the whole image
MINT(a1, CAP_TYPE_NONLIN, _start, pt_end, CAP_PERM_RWX)
CCSRRW(x0, CCSR_CPMP(0), a1)
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
mv s6, t0                      # the satp value, for the frame (s6 = x22; s10 is x26, the context's capability)

# ---------------------------------------------------------------- the frame
MINT(s9, CAP_TYPE_NONLIN, frame, frame_end, CAP_PERM_RW)
sd s6, 0(s9)                   # satp
lla t0, tbl
sd t0, 8(s9)                   # srevroot
sd x0, 56(s9)                  # options 0
MINT(a2, CAP_TYPE_LIN, ctx, ctx_end, CAP_PERM_RWX)
STC(s9, a2, 64)                # the PCC slot
li t0, 480
CINCOFFSET(t1, s9, t0)
STC(t1, x26, 0)                # the x26 slot (+64 + 16 * 26)

#ifdef VA_QUANTUM
li t0, 3000
#else
li t0, 0x100000
#endif
csrw CSUPQUANTUM, t0

# ---------------------------------------------------------------- the control: CSRUNV in legacy C is illegal (2)
lla a0, frame
li a1, 0
CSRUNV(a2, a0, a1)             # traps: the handler prints mcause (2) and steps over it

# ---------------------------------------------------------------- into scalar S
li s2, 0x1234
li t0, 0x1800                  # MPP := 01 (S)
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
CSRUNV(a2, a0, a1)             # the entry; S resumes here with a2 = the kind
mv a3, a2
li a7, 1
ecall                          # to M: the readings
lla a0, frame
li a1, 1
CSRUNV(a2, a0, a1)             # action 1: discard
mv a3, a2
li a7, 2
ecall
1: j 1b

.align 12
trap_handler:
  csrr t0, mcause
  li t1, 2
  bne t0, t1, 3f
  CAPPRINT(t0)                 # 2: the legacy-C control
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
3:
  li t1, 1
  bne a7, t1, 4f
  CAPPRINT(a3)                 # the kind: 2 (quantum variant: 1)
  CAPPRINT(s2)                 # 0x1234: the caller's registers came back
  ld t2, 16(s9)
  CAPPRINT(t2)                 # +16 kind
  ld t2, 24(s9)
  CAPPRINT(t2)                 # +24 cause
  ld t2, 32(s9)
#ifdef VA_QUANTUM
  lla t3, ctx_loop
#else
  lla t3, ctx_ecall
#endif
  sub t2, t2, t3
  CAPPRINT(t2)                 # +32 pc: 0
  ld t2, 40(s9)
  CAPPRINT(t2)                 # +40 addr: 0
  ld t2, 48(s9)
  CAPPRINT(t2)                 # +48 raw a0
  ld t2, 64(s9)
  sub t2, t2, t3
  CAPPRINT(t2)                 # the PCC slot's cursor: 0
  ld t2, 240(s9)
  CAPPRINT(t2)                 # the x11 slot
  ld t2, 8(x26)
  CAPPRINT(t2)                 # data[8]
  csrr t2, CSUPSTATUS
  CAPPRINT(t2)                 # 5 / 3
  csrr t2, CSUPCAUSE
  CAPPRINT(t2)                 # 11 / the quantum mark
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
4:
  CAPPRINT(a3)                 # 0: the discard
  csrr t0, mcause
  CAPPRINT(t0)                 # 9
  CAP_PASS(s11)

.align 6
ctx:
#ifdef VA_QUANTUM
ctx_loop:
  j ctx_loop
#else
  ld t1, 0(x26)
  sd t1, 8(x26)
  li a0, 0x55
  mv x11, t1
ctx_ecall:
  ecall
  j ctx_ecall
#endif
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
