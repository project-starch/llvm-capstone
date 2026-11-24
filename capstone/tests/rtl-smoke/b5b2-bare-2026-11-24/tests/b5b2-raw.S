# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-raw.S -- B5b-2 step 5: CLDRAW and CSTRAW, the opaque 16-byte transport (raw_tag in the lane)
#-----------------------------------------------------------------------------
#
# The reference: capstone-qemu 9289114e06 op_helper.c helper_cldraw / helper_cstraw / raw_granule_probe,
# trans_capstone.c.inc raw_moves_allowed; the arms port tests/capstone-raw-moves/probe.S. CLDRAW (custom-0 funct3 4)
# loads 128 stored bits and the memory tag into a register that grants nothing (usable tag 0, the memory tag kept as
# raw_tag); CSTRAW (funct3 5) stores them and republishes raw_tag. Trusted S or M only. A CSTRAW of a decoded (tagged)
# source raises 24 before anything is stored, ahead of the alignment check. Both are plain accesses: no capability
# check even through a tagged base. Every ordinary register write destroys raw_tag; MOVC does too.
# Departures from probe.S, each forced: the memory tag is read back with an M ldc and LCC selector 1 (no CPTAG before
# step 6); arm 10 (raw cargo as a base) runs in protected U, because in this configuration S takes a plain base (step
# 2b); arms 7 / 8 (MMIO: no taggable-memory map in simulation), 12 (default off: no property), 13-17 (translated aliases)
# and 19 (debugger) are not ported.
# Table and capabilities: b5b2-sref.S's setup. x26/s10 = {0,3} live over data, x20/s4 = x26 moved to data + 4088,
# x24/s8 = data as an integer. a1 = data + 0x800 (src), a2 = a1 + 16 (dst), a3 = a1 + 32 (dst2), integers. Traps go to
# M; the handler records mcause in x9/s1, counts in x28/t3, resumes at mepc + 4; ecall returns to M at x30/t5. Readings
# are stored at x25/s9 (S too); U holds its own in a6, a7, a0. M prints them all at the end, after their count. Type is
# LCC selector 1: 1 = a tagged NONLIN, 7 = untagged; an M ldc of a granule reads its memory tag that way.
# Readings (M unless stated):
#   M1 (tagged round trip; src = stc of x26)   1 cause of cldraw t2 <- src   2 type of t2 (raw: untagged)
#      3 cause of cstraw t2 -> dst   4 type of ldc <- dst   5 dst word 0 == src word 0   6 dst word 1 == src word 1
#   M2 (the register keeps its cargo)          7 cause of cstraw t2 -> dst2   8 type of ldc <- dst2
#   M3 (untagged payload over a tagged slot)   9 type of ldc <- dst   10 / 11 dst words == the payload
#   M4 (a scalar write destroys raw_tag)       12 cause of cstraw (t2 after addi t2, t2, 0) -> dst2   13 type of ldc
#      <- dst2   14 dst2 word 1 == 0 (the lane is gone)   15 dst2 word 0 == src word 0
#   M5 (MOVC clears raw_tag)                   16 type of ldc <- dst2 after cstraw of movc(t2)   17 dst2 word 1 ==
#      src word 1 (movc keeps an untagged lane's metadata)   18 type after cstraw of t2 after movc t2, t2
#   M6 (a decoded source)                      19 cause of cstraw x26 -> dst2   20 dst2 word 0 still 0x55 (nothing
#      stored)   21 cause of cstraw x26 -> dst2 + 8 (tagged AND misaligned: 24 first)
#   M7 (alignment)                             22 cldraw t2 <- src + 8   23 cstraw t2 -> dst + 8   24 cldraw x0 <- src + 8
#   M8 (x0 stores zeros, no tag; dst2 = stc of x26 first)   25 cause of cstraw x0 -> dst2   26 type of ldc <- dst2
#      27 / 28 dst2 words == 0
#   S (srevroot = tbl)   29 cause of cldraw <- 8(x20) (a tagged base one past its bounds: plain)   30 cause of
#      ldc <- 8(x20) (the control: rule 1 checks that base)   31 cause of cldraw src then cstraw -> dst
#   M  32 type of ldc <- dst (S's round trip)
#   U (protected; M first loads raw cargo into t2)   33 cldraw   34 cstraw   35 ld through t2 (raw cargo as a base)
#      36 traps in U
#   37 the trap count
# PRE-REGISTERED (2026-11-24, written before any run), from the reference's helpers and probe.S's expectations:
#   step 5 (b5b2-step5):
#     0 7 0 1 1 1 | 0 1 | 7 1 1 | 0 7 1 1 | 7 1 7 | 0x18 1 0x18 | 4 6 4 | 0 7 1 1 | 0 0x1c 0 | 1 | 2 2 0x18 3 | 9
#     (17 rests on the enumeration's reading of MOVC: an untagged source's metadata passes through; the reference does
#     not check that word)
#   base (eb03e7c99, step 4: custom-0 funct3 4 / 5 are illegal, so every raw site traps 2 and is skipped):
#     2 7 2 7 0 0 | 2 7 | 7 0 0 | 2 7 1 0 | 7 0 7 | 2 1 2 | 2 2 2 | 2 1 0 0 | 2 0x1c 2 | 7 | 2 2 0x18 3 | 25
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

# ---------------------------------------------------------------- M1: a tagged round trip
STC(a1, x26, 0)
li t2, 0
CLR; RAWLD(t2, a1, 0); CAUSE
TY(t2)
CLR; RAWST(t2, a2, 0); CAUSE
li x22, 0; LDC(x22, a2, 0); TY(x22)
ld t1, 0(a1); ld a4, 0(a2); EQ(t1, a4)
ld t1, 8(a1); ld a4, 8(a2); EQ(t1, a4)
# ---------------------------------------------------------------- M2: the register keeps its cargo
CLR; RAWST(t2, a3, 0); CAUSE
li x22, 0; LDC(x22, a3, 0); TY(x22)
# ---------------------------------------------------------------- M3: an untagged payload over a tagged slot
li t1, 0xfedcba9876543210
sd t1, 0(a1)
li a4, 0x13579bdf2468ace0
sd a4, 8(a1)
li t2, 0
RAWLD(t2, a1, 0)
RAWST(t2, a2, 0)
li x22, 0; LDC(x22, a2, 0); TY(x22)
ld x23, 0(a2); EQ(x23, t1)
ld x23, 8(a2); EQ(x23, a4)
# ---------------------------------------------------------------- M4: a scalar write destroys raw_tag
STC(a1, x26, 0)
li t2, 0
RAWLD(t2, a1, 0)
addi t2, t2, 0
CLR; RAWST(t2, a3, 0); CAUSE
li x22, 0; LDC(x22, a3, 0); TY(x22)
ld x23, 8(a3); EQI(x23, 0)
ld x23, 0(a3); ld t1, 0(a1); EQ(x23, t1)
# ---------------------------------------------------------------- M5: MOVC clears raw_tag
li t2, 0
RAWLD(t2, a1, 0)
li x31, 0
MOVC(x31, t2)
RAWST(x31, a3, 0)
li x22, 0; LDC(x22, a3, 0); TY(x22)
ld x23, 8(a3); ld t1, 8(a1); EQ(x23, t1)
RAWLD(t2, a1, 0)
MOVC(t2, t2)
RAWST(t2, a3, 0)
li x22, 0; LDC(x22, a3, 0); TY(x22)
# ---------------------------------------------------------------- M6: a decoded source
li t1, 0x55
sd t1, 0(a3)
sd x0, 8(a3)
CLR; RAWST(x26, a3, 0); CAUSE
ld x23, 0(a3); EQI(x23, 0x55)
CLR; RAWST(x26, a3, 8); CAUSE
# ---------------------------------------------------------------- M7: alignment
CLR; RAWLD(t2, a1, 8); CAUSE
CLR; RAWST(t2, a2, 8); CAUSE
CLR; RAWLD(x0, a1, 8); CAUSE
# ---------------------------------------------------------------- M8: x0 stores zeros and no tag
STC(a3, x26, 0)
CLR; RAWST(x0, a3, 0); CAUSE
li x22, 0; LDC(x22, a3, 0); TY(x22)
ld x23, 0(a3); EQI(x23, 0)
ld x23, 8(a3); EQI(x23, 0)
# ---------------------------------------------------------------- S (srevroot = tbl)
TO_S(s_raw, m_after_s)
m_after_s:
li x22, 0; LDC(x22, a2, 0); TY(x22)
# ---------------------------------------------------------------- U (protected): raw moves refused; raw cargo is no base
li t2, 0
RAWLD(t2, a1, 0)
li a6, 0
li a7, 0
li a0, 0
mv x31, x28
TO_U(u_raw, m_after_u)
m_after_u:
mv a5, a6; REC
mv a5, a7; REC
mv a5, a0; REC
sub a5, x28, x31; REC
mv a5, x28; REC

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
