# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-runv2.S  (vp2 item D, step D2: the resumable kinds 3 and 4, actions 0 and 2, the views, the frame checks, 2026-10-17)
#-----------------------------------------------------------------------------
#
# The setup of va-runv.S with options 3 and one more page, `late`, left unmapped (its L0 entry 0) with the s4 slot a
# NONLIN RW over it. The context: a load through s4 (a page fault, kind 4), then after the resume a store of the loaded
# word into data[0], a service ECALL with a0 = 7 (kind 3), after the reply a store of a0 into data[8], then an illegal
# instruction (kind 2). The S-mode supervisor runs the actions and ECALLs into M for the readings; M maps the page
# before the first resume and writes the reply 0x99 into the a0 slot (+224) before the second. Every refused action
# (2 on a kind-4 pause, 0 on a kind-3 pause, 0 and 2 on the terminal slot) traps 29 in S, read by the handler. Then
# four bad frames, one check each (satp not Sv39, srevroot 0, options 4, the PCC slot untagged), each refused as the
# terminal event {2, 29, the PCC slot's cursor, 0} with rd 2 (the semantics owner's ruling), each discarded after.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order), read 2026-10-17 exactly so:
#   2 (the legacy-C control);
#   R1 (the page fault): rd 4; +16 4; +24 13; +32 minus ctx_ld 0; +40 minus late 0; +48 0; the a0 view at +576 0;
#     csupstatus 5;  then the refused action 2: 29;
#   R2 (the service ECALL): rd 3; +16 3; +24 11; +32 minus ctx_ecall 0; +48 7; the views +576 7, +584 0, +632 0;
#     data[0] 0x42;  then the refused action 0: 29;
#   R3 (the illegal instruction after the reply): rd 2; +16 2; +24 2; +32 minus ctx_bad 0; data[8] 0x99;
#     then the refused actions 0 and 2: 29, 29;  the discard: 0;  the re-minted PCC slot's cursor minus ctx: 0;
#   the four refusals, each: rd 2; the PCC slot's low word (ctx for the first three, 0x1234 for the untagged slot);
#     +16 2; +24 29; +32 minus ctx 0 (the untagged-PCC case prints +32 raw: 0x1234); the discard 0.
#   (A first cut re-stored the PCC with STC from the same register: a LINEAR capability moves on a store, so the
#   slot read back 0; the handler mints it again. The views' copy loop with an index computed inside the call's
#   arguments generated code that never wrote; it is unrolled in the switcher.)

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSRUNV(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x24, rd, rs1, rs2

#define CSUPQUANTUM 0x7C3
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1

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

# ---------------------------------------------------------------- data and late, and the context's capabilities (node 2)
MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)
sd x0, 0(x26)
sd x0, 8(x26)
MINT(s4, CAP_TYPE_NONLIN, late, late_end, CAP_PERM_RW)
li t0, 0x42
sd t0, 0(s4)

# ---------------------------------------------------------------- page tables as va-runv.S; L0[late] invalid
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
# L0[late] -> invalid; s5 keeps the entry's capability for the handler's mapping
lla t0, late
li t1, 0x80000000
sub t0, t0, t1
srli t0, t0, 12
slli t0, t0, 3
add t0, t0, a7
CINCOFFSET(s5, s7, t0)
sd x0, 0(s5)
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
mv s6, t0                      # the satp value (s6 = x22)

# ---------------------------------------------------------------- the frame
MINT(s9, CAP_TYPE_NONLIN, frame, frame_end, CAP_PERM_RW)
sd s6, 0(s9)                   # satp
lla t0, tbl
sd t0, 8(s9)                   # srevroot
li t0, 3
sd t0, 56(s9)                  # options: resumable ECALL and page faults
MINT(s3, CAP_TYPE_LIN, ctx, ctx_end, CAP_PERM_RWX)
STC(s9, s3, 64)                # the PCC slot (s3 keeps a copy for the refusal frames)
li t0, 480
CINCOFFSET(t1, s9, t0)
STC(t1, x26, 0)                # the x26 slot
li t0, 384
CINCOFFSET(t1, s9, t0)
STC(t1, s4, 0)                 # the s4 slot (+64 + 16 * 20)
li t0, 0x100000
csrw CSUPQUANTUM, t0

# ---------------------------------------------------------------- the control: CSRUNV in legacy C is illegal (2)
lla a0, frame
li a1, 0
CSRUNV(a2, a0, a1)

# ---------------------------------------------------------------- into scalar S
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
CSRUNV(a2, a0, a1)             # entry; the page fault pauses it: a2 = 4
mv a3, a2
li a7, 1
ecall                          # R1; M maps the page
li a1, 2
CSRUNV(a2, a0, a1)             # action 2 on a kind-4 pause: refused, 29
li a1, 0
CSRUNV(a2, a0, a1)             # resume: the service ECALL pauses it: a2 = 3
mv a3, a2
li a7, 2
ecall                          # R2; M writes the reply
li a1, 0
CSRUNV(a2, a0, a1)             # action 0 on a kind-3 pause: refused, 29
li a1, 2
CSRUNV(a2, a0, a1)             # the reply consumed; the illegal instruction ends it: a2 = 2
mv a3, a2
li a7, 3
ecall                          # R3
li a1, 0
CSRUNV(a2, a0, a1)             # terminal: refused, 29
li a1, 2
CSRUNV(a2, a0, a1)             # terminal: refused, 29
li a1, 1
CSRUNV(a2, a0, a1)             # discard: 0
mv a3, a2
li a7, 4
ecall                          # M prepares the first bad frame
li a7, 5
3:
li a1, 0
CSRUNV(a2, a0, a1)             # a bad frame: the terminal event, rd 2
mv a3, a2
li a1, 1
CSRUNV(a4, a0, a1)             # discard: 0
ecall                          # the readings; M prepares the next
addi a7, a7, 1
j 3b

.align 12
trap_handler:
  csrr t0, mcause
  li t1, 9
  beq t0, t1, 5f
  CAPPRINT(t0)                 # 2 (the control) or 29 (a refused action)
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
5:
  li t1, 1
  bne a7, t1, 6f
  CAPPRINT(a3)                 # 4
  ld t2, 16(s9)
  CAPPRINT(t2)                 # 4
  ld t2, 24(s9)
  CAPPRINT(t2)                 # 13
  ld t2, 32(s9)
  lla t3, ctx_ld
  sub t2, t2, t3
  CAPPRINT(t2)                 # 0
  ld t2, 40(s9)
  lla t3, late
  sub t2, t2, t3
  CAPPRINT(t2)                 # 0
  ld t2, 48(s9)
  CAPPRINT(t2)                 # 0
  ld t2, 576(s9)
  CAPPRINT(t2)                 # 0
  csrr t2, CSUPSTATUS
  CAPPRINT(t2)                 # 5
  # map `late`
  lla t1, late
  srli t1, t1, 12
  slli t1, t1, 10
  ori t1, t1, PTE_LEAF_RWX
  sd t1, 0(s5)
  sfence.vma
  j 9f
6:
  li t1, 2
  bne a7, t1, 7f
  CAPPRINT(a3)                 # 3
  ld t2, 16(s9)
  CAPPRINT(t2)                 # 3
  ld t2, 24(s9)
  CAPPRINT(t2)                 # 11
  ld t2, 32(s9)
  lla t3, ctx_ecall
  sub t2, t2, t3
  CAPPRINT(t2)                 # 0
  ld t2, 48(s9)
  CAPPRINT(t2)                 # 7
  ld t2, 576(s9)
  CAPPRINT(t2)                 # 7
  ld t2, 584(s9)
  CAPPRINT(t2)                 # 0: the a1 view
  ld t2, 632(s9)
  CAPPRINT(t2)                 # 0: the a7 view
  ld t2, 0(x26)
  CAPPRINT(t2)                 # 0x42
  li t2, 0x99
  sd t2, 224(s9)               # the reply
  j 9f
7:
  li t1, 3
  bne a7, t1, 8f
  CAPPRINT(a3)                 # 2
  ld t2, 16(s9)
  CAPPRINT(t2)                 # 2
  ld t2, 24(s9)
  CAPPRINT(t2)                 # 2
  ld t2, 32(s9)
  lla t3, ctx_bad
  sub t2, t2, t3
  CAPPRINT(t2)                 # 0
  ld t2, 8(x26)
  CAPPRINT(t2)                 # 0x99
  j 9f
8:
  li t1, 4
  bne a7, t1, 10f
  CAPPRINT(a3)                 # 0: the discard
  # a LINEAR capability moves on a store: mint the PCC again (through t-registers: a0 holds the frame, a4 the discard)
  CAPCREATE(s3); li t0, CAP_TYPE_LIN; CAPTYPE(s3, t0); lla t0, ctx; lla t1, ctx_end; CAPBOUND(s3, t0, t1); li t0, CAP_PERM_RWX; CAPPERM(s3, t0)
  STC(s9, s3, 64)              # a fresh PCC for the refusals
  ld t2, 64(s9)
  lla t3, ctx
  sub t2, t2, t3
  CAPPRINT(t2)                 # 0: the slot's low word minus ctx after the STC
  sd x0, 0(s9)                 # (a) satp 0
  j 9f
10:
  CAPPRINT(a3)                 # 2
  ld t2, 64(s9)
  CAPPRINT(t2)                 # the PCC slot's low word (ctx; 0x1234 for the untagged slot)
  ld t2, 16(s9)
  CAPPRINT(t2)                 # 2
  ld t2, 24(s9)
  CAPPRINT(t2)                 # 29
  ld t2, 32(s9)
  li t1, 8
  beq a7, t1, 11f
  lla t3, ctx
  sub t2, t2, t3
11:
  CAPPRINT(t2)                 # 0 (the PCC cursor is ctx) / 0x1234 raw for the untagged slot
  CAPPRINT(a4)                 # 0: the discard
  li t1, 5
  bne a7, t1, 12f
  sd s6, 0(s9)                 # satp back; (b) srevroot 0
  sd x0, 8(s9)
  j 9f
12:
  li t1, 6
  bne a7, t1, 13f
  lla t2, tbl
  sd t2, 8(s9)                 # srevroot back; (c) options 4
  li t2, 4
  sd t2, 56(s9)
  j 9f
13:
  li t1, 7
  bne a7, t1, 14f
  li t2, 3
  sd t2, 56(s9)                # options back; (d) the PCC slot untagged
  li t2, 0x1234
  sd t2, 64(s9)
  j 9f
14:
  CAP_PASS(s11)
9:
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 6
ctx:
ctx_ld:
  ld t1, 0(s4)
  sd t1, 0(x26)
  li a0, 7
ctx_ecall:
  ecall
  sd a0, 8(x26)
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
late:
  .zero 4096
late_end:
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
