# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp.

#*****************************************************************************
# va-cscheck-mon.S  (MEASUREMENT ONLY: the monitor's CSCHECKR/W and CSKINDS emulation for the native route on
# C.5a/E5, 2026-11-08; to be run by the RTL lane, as va-csmint-mon was)
#-----------------------------------------------------------------------------
#
# The real path, the same shape as va-csmint-mon: S issues the instruction -> cause 2 -> an M trap entry that builds the
# monitor's frame (S's x[n] at frame + 8n, S's sp in CSCRATCH) -> cscheck_dispatch from capstone-sbi's
# capstone_cscheck.S (native/cscheck-emu, #included), then csmint_dispatch for CSMINT -> rd written into the frame.
# S builds a native context frame page (cframe) itself: +0 satp, +8 srevroot, +2336 the epoch, and the slots at
# +64 + 16r filled by CSMINT (the emulated one), so the capabilities have real nodes in the flat RECYCLE table.
#
# Per row S presets a2 = -1, fills the 48-byte descriptor, issues the instruction, then ecalls; M prints S's a2 (rd),
# descriptor[4] and descriptor[5] (the fingerprint 0x4353434845 on every check the emulation handled).
# Pre-registered (VAs = PAs, identity map; nm gives the symbols):
#   M0  CSMINT D1 {dbuf, dbuf+256, RW, NONLIN} into cframe x10 slot    rd = the ancestor id (4 with this table)
#   M1  CSMINT D4 {dbuf+512, dbuf+768, RO, NONLIN} into the x12 slot   6
#   M2  CSMINT D6 {dbuf, dbuf+256, RW, LINEAR} into the x13 slot       8
#   1  CSKINDS(cframe)                                                  0x3400 (x10, x12, x13; 13 reported, never loaded)
#   2  CSCHECKR epoch 7 slot 10 offset 16 bytes 32                       0, dbuf+16, FP
#   3  CSCHECKW the same                                                 0, dbuf+16, FP
#   4  CSCHECKR offset 240 bytes 32 (16 bytes left)                      28, 0, FP
#   5  CSCHECKR offset 256 bytes 0 (the end, zero bytes)                 0, dbuf+256, FP
#   6  CSCHECKR epoch 8 (frame says 7)                                   29, 0, FP
#   7  CSCHECKR slot 11 (hi word 0)                                      24, 0, FP
#   8  CSCHECKR reserved word 1                                          29, 0, FP
#   9  CSCHECKW slot 12 (read-only)                                      27, 0, FP
#  10  CSCHECKR slot 12                                                  0, dbuf+512, FP
#  11  CSCHECKR slot 13 (LINEAR: refused before any load)                26, 0, FP
#  12  CSCHECKR frame + 16 (misaligned frame)                            29, 0, 0 (refused before the descriptor)
#  13  M clears record VALID of slot 10's id, then CSCHECKR slot 10      25, 0, FP (24 if the RTL's LDC drops a dead
#      node's tag itself; either is a refusal, the number says which mechanism fired)
#  final: slot 12 read back by M: LCC type 1 (NONLIN, still there after rows 9-10's loads); slot 13's hi word
#         non-zero (the LINEAR capability was NOT moved out by row 11)
# Any 0xbad1/0xbad2 is a failure of the arm, not a reading. FP = 0x4353434845.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSMINT(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x50, rd, rs1, rs2
#define CSCHECKR(rd, rs1, rs2) .insn r 0x5B, 0x1, 0x25, rd, rs1, rs2
#define CSCHECKW(rd, rs1, rs2) .insn r 0x5B, 0x1, 0x26, rd, rs1, rs2
#define CSKINDS(rd, rs1)      .insn r 0x5B, 0x1, 0x27, rd, rs1, x0
#define SREVROOT 0x5C1
#define PTE_S_RWX   0xCF
#define PTE_PTR     0x01
#define RAM_BASE 0x80000000

.option norvc

.section .text.init
.globl _start
_start:

li gp, 1
la t0, m_entry
csrw mtvec, t0
csrw mcause, x0

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)
#define PHYS(REG, AREG) li t6, RAM_BASE; sub t6, AREG, t6; CINCOFFSET(REG, s3, t6)

MINT(s3, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RW)
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
MINT(a2, CAP_TYPE_NONLIN, frame, frame_end, CAP_PERM_RW)
CCSRRW(x0, CCSR_CSCRATCH, a2)

# ---------------------------------------------------------------- the table: RECYCLE, capacity 64, next 3, record 2 valid
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
lla t0, tbl
csrw SREVROOT, t0

# ---------------------------------------------------------------- Sv39 tables: identity, 4 KiB supervisor pages
lla t0, pt_l0
PHYS(s4, t0)
li t1, RAM_BASE
li t2, 0
1:
srli t3, t1, 12
slli t3, t3, 10
ori t3, t3, PTE_S_RWX
slli t4, t2, 3
CINCOFFSET(t0, s4, t4)
sd t3, 0(t0)
li t4, 0x1000
add t1, t1, t4
addi t2, t2, 1
li t4, 512
bne t2, t4, 1b
lla t0, pt_l1
PHYS(t1, t0)
lla t2, pt_l0
srli t2, t2, 12
slli t2, t2, 10
ori t2, t2, PTE_PTR
sd t2, 0(t1)
lla t0, pt_root
PHYS(t1, t0)
lla t2, pt_l1
srli t2, t2, 12
slli t2, t2, 10
ori t2, t2, PTE_PTR
sd t2, 16(t1)
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
MINT(a1, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RWX)
CCSRRW(x0, CCSR_CPMP(0), a1)
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma

li t0, 0x1800
csrc mstatus, t0
li t0, 0x0800
csrs mstatus, t0
lla t0, s_code
csrw mepc, t0
mret

# ================================================================ S
.align 12
s_code:
la t0, s_trap
csrw stvec, t0
# the context frame's words the check reads: satp, srevroot, the epoch
lla s1, cframe
csrr t0, satp
sd t0, 0(s1)
csrr t0, SREVROOT
sd t0, 8(s1)
li t0, 2336
add t1, s1, t0
li t0, 7
sd t0, 0(t1)
#define REPORT(K) li a7, K; ecall
# CSMINT descriptors (v7: base, end, perms | NONLIN << 8, cursor, second slot, reserved)
#define MDESC(B, E, P) lla t0, mdesc; lla t1, dbuf; li t2, B; add t2, t2, t1; sd t2, 0(t0); sd t2, 24(t0); \
  li t2, E; add t2, t2, t1; sd t2, 8(t0); li t2, P; sd t2, 16(t0); sd x0, 32(t0); sd x0, 40(t0)
#define MROW(SLOTREG, B, E, P) MDESC(B, E, P); li a2, -1; addi a0, s1, 64 + 16 * SLOTREG; lla a1, mdesc; CSMINT(a2, a0, a1)
MROW(10, 0, 256, 0x106)        # M0: RW NONLIN
REPORT(50)
MROW(12, 512, 768, 0x104)      # M1: RO NONLIN
REPORT(50)
MROW(13, 0, 256, 0x006)        # M2: RW LINEAR
REPORT(50)
# CSCHECK descriptor (48 bytes, 16-aligned): epoch, slot, offset, bytes, out, reserved
#define CDESC(EP, SL, OFF, BY, RES) lla t0, cdesc; li t1, EP; sd t1, 0(t0); li t1, SL; sd t1, 8(t0); li t1, OFF; sd t1, 16(t0); \
  li t1, BY; sd t1, 24(t0); sd x0, 32(t0); li t1, RES; sd t1, 40(t0)
li a2, -1                      # 1
CSKINDS(a2, s1)
REPORT(51)
CDESC(7, 10, 16, 32, 0)        # 2
li a2, -1
lla a1, cdesc
CSCHECKR(a2, s1, a1)
REPORT(52)
CDESC(7, 10, 16, 32, 0)        # 3
li a2, -1
CSCHECKW(a2, s1, a1)
REPORT(52)
CDESC(7, 10, 240, 32, 0)       # 4
li a2, -1
CSCHECKR(a2, s1, a1)
REPORT(52)
CDESC(7, 10, 256, 0, 0)        # 5
li a2, -1
CSCHECKR(a2, s1, a1)
REPORT(52)
CDESC(8, 10, 16, 32, 0)        # 6
li a2, -1
CSCHECKR(a2, s1, a1)
REPORT(52)
CDESC(7, 11, 16, 32, 0)        # 7
li a2, -1
CSCHECKR(a2, s1, a1)
REPORT(52)
CDESC(7, 10, 16, 32, 1)        # 8
li a2, -1
CSCHECKR(a2, s1, a1)
REPORT(52)
CDESC(7, 12, 16, 32, 0)        # 9
li a2, -1
CSCHECKW(a2, s1, a1)
REPORT(52)
CDESC(7, 12, 0, 32, 0)         # 10
li a2, -1
CSCHECKR(a2, s1, a1)
REPORT(52)
CDESC(7, 13, 16, 32, 0)        # 11
li a2, -1
CSCHECKR(a2, s1, a1)
REPORT(52)
CDESC(7, 10, 16, 32, 0)        # 12: a misaligned frame
li a2, -1
addi a0, s1, 16
CSCHECKR(a2, a0, a1)
REPORT(52)
li a7, 23                      # 13: M clears slot 10's record VALID
ecall
CDESC(7, 10, 16, 32, 0)
li a2, -1
CSCHECKR(a2, s1, a1)
REPORT(52)
li a7, 99
ecall
1: j 1b

.align 4
s_trap:                        # an injected refusal would land here: record it, step over
csrr a5, scause
csrr a6, stval
csrr t0, sepc
addi t0, t0, 4
csrw sepc, t0
sret

# ================================================================ M
.align 12
m_entry:
CCSRRW(sp, CCSR_CSCRATCH, sp)
sd x1, 8(sp)
sd x3, 24(sp)
sd x4, 32(sp)
sd x5, 40(sp)
sd x6, 48(sp)
sd x7, 56(sp)
sd x8, 64(sp)
sd x9, 72(sp)
sd x10, 80(sp)
sd x11, 88(sp)
sd x12, 96(sp)
sd x13, 104(sp)
sd x14, 112(sp)
sd x15, 120(sp)
sd x16, 128(sp)
sd x17, 136(sp)
sd x18, 144(sp)
sd x19, 152(sp)
sd x20, 160(sp)
sd x21, 168(sp)
sd x22, 176(sp)
sd x23, 184(sp)
sd x24, 192(sp)
sd x25, 200(sp)
sd x26, 208(sp)
sd x27, 216(sp)
sd x28, 224(sp)
sd x29, 232(sp)
sd x30, 240(sp)
sd x31, 248(sp)
csrr t0, mcause
li t1, 2
bne t0, t1, m_not_illegal
MOVC(a0, sp)
call cscheck_dispatch
bnez a0, m_return
MOVC(a0, sp)
call csmint_dispatch
bnez a0, m_return
li t0, 0xbad1
CAPPRINT(t0)
j m_skip
m_not_illegal:
li t1, 9
beq t0, t1, m_ecall
li t0, 0xbad2
CAPPRINT(t0)
csrr t0, mcause
CAPPRINT(t0)
j m_skip
m_ecall:
MINT(s3, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RW)
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
ld t0, 136(sp)                 # S's a7
li t1, 50
beq t0, t1, m_rd
li t1, 51
beq t0, t1, m_rd
li t1, 52
beq t0, t1, m_check
li t1, 23
beq t0, t1, m_kill
li t1, 99
beq t0, t1, m_final
li t0, 0xbad2
CAPPRINT(t0)
j m_skip
m_rd:
ld t2, 96(sp)
CAPPRINT(t2)                   # S's a2
j m_skip
m_check:
ld t2, 96(sp)
CAPPRINT(t2)                   # rd
lla t0, cdesc
PHYS(t1, t0)
ld t2, 32(t1)
CAPPRINT(t2)                   # descriptor[4]
ld t2, 40(t1)
CAPPRINT(t2)                   # descriptor[5]
j m_skip
m_kill:                        # clear VALID of slot 10's record: its id from the slot's hi word
lla t0, cframe
addi t0, t0, 64 + 16 * 10
PHYS(t1, t0)
ld t2, 8(t1)
srli t2, t2, 34
li t3, 0xfffff
and t2, t2, t3                 # the index (20 bits)
slli t2, t2, 4
addi t2, t2, 8
CINCOFFSET(t1, s8, t2)
ld t3, 0(t1)
li t4, 1
slli t4, t4, 60
not t4, t4
and t3, t3, t4
sd t3, 0(t1)
j m_skip
m_final:
lla t0, cframe
addi t0, t0, 64 + 16 * 12
PHYS(t1, t0)
LDC(t3, t1, 0)
LCC(t2, t3, x1)
CAPPRINT(t2)                   # 1: NONLIN, still in the slot
li t3, 0
lla t0, cframe
addi t0, t0, 64 + 16 * 13
PHYS(t1, t0)
ld t2, 8(t1)
CAPPRINT(t2)                   # the LINEAR slot's hi word, non-zero: row 11 did not move it
CAP_PASS(s11)
m_skip:
csrr t0, mepc
addi t0, t0, 4
csrw mepc, t0
m_return:
ld x1, 8(sp)
ld x3, 24(sp)
ld x4, 32(sp)
ld x5, 40(sp)
ld x6, 48(sp)
ld x7, 56(sp)
ld x8, 64(sp)
ld x9, 72(sp)
ld x10, 80(sp)
ld x11, 88(sp)
ld x12, 96(sp)
ld x13, 104(sp)
ld x14, 112(sp)
ld x15, 120(sp)
ld x16, 128(sp)
ld x17, 136(sp)
ld x18, 144(sp)
ld x19, 152(sp)
ld x20, 160(sp)
ld x21, 168(sp)
ld x22, 176(sp)
ld x23, 184(sp)
ld x24, 192(sp)
ld x25, 200(sp)
ld x26, 208(sp)
ld x27, 216(sp)
ld x28, 224(sp)
ld x29, 232(sp)
ld x30, 240(sp)
ld x31, 248(sp)
CCSRRW(sp, CCSR_CSCRATCH, sp)
mret

# ================================================================ the monitor's routines, from capstone-sbi native/cscheck-emu
#include "capstone_cscheck.S"
#include "capstone_csmint.S"

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
pt_root:
  .zero 4096
pt_l1:
  .zero 4096
pt_l0:
  .zero 4096
tbl:
  .zero 4096
tbl_end:
mdesc:
  .zero 64
cdesc:
  .zero 4032
cframe:
  .zero 4096
dbuf:
  .zero 4096
frame:
  .zero 256
frame_end:
  TEST_DATA
RVTEST_DATA_END
.align 12
ram_end:
