# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-rawlcc.S -- B5b-2 step 5: a capability instruction reading a raw register as rs1 leaves its raw_tag (fix 2)
#-----------------------------------------------------------------------------
#
# The reference (capstone-qemu 9289114e06): LCC and MOVC write nothing to rs1, so a raw register keeps its retained
# tag through them. On fcd169dc5 / 39e83f3be every FLU / DYN instruction hands rs1 back at commit through cap_lane,
# which zeroes the lane's bit 65 (raw_tag) of an untagged rs1: b5b2-raw readings 4 and 8 read untagged. Found by that
# run and isolated by an uncommitted matched pair (arms E / F below, same tree). The semantics owner ruled it a defect
# to fix (no divergence from the reference). Fix 2: per scoreboard entry, whether rs1 was tagged at issue; commit drops
# the rs1 write of an untagged rs1, and such an entry is no rs1 forwarding source.
# b5b2-raw.S's setup (via b5b2-sref.S): x26/s10 = {0,3} live over data, x24/s8 = data as an integer; a1 = data + 0x800
# (src, holds stc of x26), a2 / a3 = a1 + 16 / + 32. Each arm: cldraw t2 <- src, the instruction under test, cstraw
# t2 -> dst, then an M ldc of dst read with LCC selector 1 (1 = the memory tag was republished, 7 = it was lost).
#   E  lcc t1, t2, selector 1 between the cldraw and the cstraw (a DYN op with rs1 = the raw register)
#   F  E's control: the same lcc reading x26 instead
#   G  movc x31, t2 then cstraw t2 back to back (MOVC is an rs1 forwarding source: the forwarding gate)
#   H  movc x31, t2, a delay, then cstraw t2 (the commit gate: the register file)
# PRE-REGISTERED (2026-11-24, written before any run):
#   before the fix (b5b2-step5 39e83f3be core):  7 1 7 7
#   with fix 2:                                   1 1 1 1
# E and F already ran once as an uncommitted diagnostic on fcd169dc5 (7, 1); G and H have not run.
# B5b-2 (2026-11-24, the semantics owner's ruling): uses the capcreate family, to be ported to CSMINT.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT    0x5C1

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

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

#define RAWLD(rd, base, off) .insn i 0x0b, 4, rd, base, off
#define RAWST(rs, base, off) .insn s 0x0b, 5, rs, off(base)
#define REC       sd a5, 0(x25); addi x25, x25, 8
#define TY(R)     li a5, 0; LCC(a5, R, x1); REC
#define EQ(R, V)  sub a5, R, V; seqz a5, a5; REC
#define EQI(R, I) li a5, I; sub a5, R, a5; seqz a5, a5; REC
#define CAUSE     mv a5, x9; REC
#define TO_S(LBL, BACK) li t0, 0x1800; csrc mstatus, t0; li t0, 0x800; csrs mstatus, t0; lla t0, LBL; csrw mepc, t0; lla x30, BACK; mret
#define TO_U(LBL, BACK) li t0, 0x1800; csrc mstatus, t0; lla t0, LBL; csrw mepc, t0; lla x30, BACK; mret

li t1, 0x800
add a1, x24, t1                # src  = data + 0x800
addi a2, a1, 16                # dst
addi a3, a1, 32                # dst2
lla t0, tbl
csrw SREVROOT, t0

# ---------------------------------------------------------------- the four arms
STC(a1, x26, 0)
# E
sd x0, 0(a2); sd x0, 8(a2)
li t2, 0
RAWLD(t2, a1, 0)
li t1, 0; LCC(t1, t2, x1)
RAWST(t2, a2, 0)
li x22, 0; LDC(x22, a2, 0); TY(x22)
# F
sd x0, 0(a3); sd x0, 8(a3)
li t2, 0
RAWLD(t2, a1, 0)
li t1, 0; LCC(t1, x26, x1)
RAWST(t2, a3, 0)
li x22, 0; LDC(x22, a3, 0); TY(x22)
# G
sd x0, 0(a2); sd x0, 8(a2)
li t2, 0
RAWLD(t2, a1, 0)
li x31, 0
MOVC(x31, t2)
RAWST(t2, a2, 0)
li x22, 0; LDC(x22, a2, 0); TY(x22)
# H
sd x0, 0(a3); sd x0, 8(a3)
li t2, 0
RAWLD(t2, a1, 0)
li x31, 0
MOVC(x31, t2)
li t1, 20
2: addi t1, t1, -1
bnez t1, 2b
RAWST(t2, a3, 0)
li x22, 0; LDC(x22, a3, 0); TY(x22)
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
csrw SREVROOT, x0
RVTEST_PASS

.align 4
s_raw:
  CLR; RAWLD(t2, x20, 8); CAUSE
  CLR; LDC(x22, x20, 8); CAUSE
  li t2, 0
  CLR; RAWLD(t2, a1, 0); RAWST(t2, a2, 0); CAUSE
  ecall
.align 4
u_raw:
  # U records in a6 / a7 / a0 only: a store through x25, an integer base, would itself raise 24 in protected U
  CLR; RAWLD(x22, a1, 0); mv a6, x9
  CLR; RAWST(t2, a2, 0); mv a7, x9
  CLR; ld x23, 0(t2); mv a0, x9
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
.align 6
refbox:
  .zero 64
.align 6
res:
  .zero 512
  TEST_DATA
RVTEST_DATA_END
