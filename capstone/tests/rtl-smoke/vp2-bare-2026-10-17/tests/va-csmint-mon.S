# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-csmint-mon.S  (vp2: the monitor's csmint path, 2026-10-17; the board lane's, run by the RTL lane on bf3db3175+)
#-----------------------------------------------------------------------------
#
# The real path, not only the routine: S issues csmint -> cause 2 -> an M trap entry that builds the MONITOR'S frame
# (S's x[n] at frame + 8n, S's sp parked in CSCRATCH, as capstone-sbi's _cap_trap_entry) -> csmint_dispatch from
# capstone-sbi's capstone_csmint.S, #included here from the monitor source at a pinned hash (checked by the build:
# README.md) -> rd written into the frame and the frame restored, or the refusal injected into S, whose OWN stvec handler
# records scause and stval and steps over the instruction. S reads rd itself; M reads the slot back physically at the end.
# va-csmint-emu (the routine alone, 31/32 as pre-registered and the 32nd a wrong prediction) is the positive control for
# the mint; this arm is what exercises the frame, the dispatch and the injection.
#
# Per row S presets a2 = a5 = a6 = -1, issues csmint(a2, a0 = slot, a1 = desc), then ecalls; M prints S's a2 (rd), a5
# (scause, or -1 if nothing was injected) and a6 (stval). Pre-registered (VAs symbolic; nm on the built ELF gives them):
#   1  D1 into slotA                        4, -1, -1          (child 3, ancestor 4)
#   2  descriptor misaligned (desc+4)       -1, 4, desc+4
#   3  slot misaligned (slotB+8)            -1, 6, slotB+8
#   4  base >= end (D2)                     -1, 29, 0
#   5  unrepresentable (D3)                 -1, 29, 0
#   6  perms > 15 (D5)                      -1, 29, 0
#   7  slot PTE without D (slotnod)         -1, 15, slotnod
#   8  descriptor in a page without A       -1, 13, descnoa    (the monitor walks the descriptor; no MPRV)
#   9  rs1 = sp                             -1, 2, the instruction word (csmint a2, sp, a1)
#  10  D4 into slotA again                  6, -1, -1          (child 5, ancestor 6)
#  11  next-unused 63 (one free identity)   -1, 30, 0          (the routine's pre-check, before the window)
#  12  free head 0 with free count 1: the pre-check passes (64 - 63 + 1 = 2), CSNODE refuses the header (corrupt)
#                                          -1, 29, 0          (CSNODE's own fault, caught by the window vector)
#  final: slotA read back: type 0 (LCC's spec numbering: LINEAR), dbuf+512, dbuf+768, dbuf+512 (cursor), 4;
#         slotB 0 0; slotnod 0; the table's next-unused 63 and free word 0x100000000 (rows 11 and 12 wrote nothing)
# 47 values, then pass. Any 0xbad1/0xbad2 (an unexpected trap or ecall) is a failure of the arm, not a reading.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSMINT(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x50, rd, rs1, rs2
#define SREVROOT 0x5C1
#define PTE_S_RWX   0xCF   /* V R W X A D, supervisor */
#define PTE_S_NOD   0x47   /* V R W A: no D */
#define PTE_S_NOA   0x87   /* V R W D: no A */
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
#define PHYS(REG, AREG) lla t6, _start; sub t6, AREG, t6; CINCOFFSET(REG, s3, t6)

MINT(s3, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RW)     # M's RAM capability (setup and the final read-back only)
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)        # the table
MINT(a2, CAP_TYPE_NONLIN, frame, frame_end, CAP_PERM_RW)    # the trap frame, parked in CSCRATCH
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

# ---------------------------------------------------------------- descriptors {base, end, perms}
#define DESC(LBL, OFF, B, E, P) lla t0, LBL; addi t0, t0, OFF; PHYS(t1, t0); \
  lla t2, dbuf; li t3, B; add t3, t3, t2; sd t3, 0(t1); li t3, E; add t3, t3, t2; sd t3, 8(t1); li t3, P; sd t3, 16(t1)
DESC(desc, 0,   0,   256,  CAP_PERM_RW)     # D1
DESC(desc, 32,  256, 0,    CAP_PERM_RW)     # D2: base >= end
DESC(desc, 64,  8,   8200, CAP_PERM_RW)     # D3: unrepresentable
DESC(desc, 96,  512, 768,  CAP_PERM_RO)     # D4
DESC(desc, 128, 0,   256,  16)              # D5: perms > 15
DESC(descnoa, 0, 0,  256,  CAP_PERM_RW)     # in a page whose PTE lacks A

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
#define L0_SET(LBL, FLAGS) lla t1, LBL; li t2, RAM_BASE; sub t2, t1, t2; srli t2, t2, 12; slli t2, t2, 3; \
  CINCOFFSET(t0, s4, t2); srli t3, t1, 12; slli t3, t3, 10; ori t3, t3, FLAGS; sd t3, 0(t0)
L0_SET(slotnod, PTE_S_NOD)
L0_SET(descnoa, PTE_S_NOA)
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

# ---------------------------------------------------------------- drop to S
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
#define PRE li a2, -1; li a5, -1; li a6, -1
#define ROW(SLOT, DLBL, DOFF) PRE; lla a0, SLOT; lla a1, DLBL; addi a1, a1, DOFF; CSMINT(a2, a0, a1)
#define REPORT(K) li a7, K; ecall
ROW(slotA, desc, 0)            # 1
REPORT(1)
ROW(slotA, desc, 4)            # 2: descriptor misaligned
REPORT(2)
PRE                            # 3: slot misaligned
lla a0, slotB
addi a0, a0, 8
lla a1, desc
CSMINT(a2, a0, a1)
REPORT(3)
ROW(slotB, desc, 32)           # 4
REPORT(4)
ROW(slotB, desc, 64)           # 5
REPORT(5)
ROW(slotB, desc, 128)          # 6
REPORT(6)
ROW(slotnod, desc, 0)          # 7
REPORT(7)
ROW(slotB, descnoa, 0)         # 8
REPORT(8)
PRE                            # 9: rs1 = sp
lla a1, desc
CSMINT(a2, sp, a1)
REPORT(9)
ROW(slotA, desc, 96)           # 10
REPORT(10)
li a7, 21                      # next-unused 63
ecall
ROW(slotB, desc, 0)            # 11
REPORT(11)
li a7, 22                      # free head 0, free count 1
ecall
ROW(slotB, desc, 0)            # 12
REPORT(12)
li a7, 99
ecall
1: j 1b

# S's own trap handler: record the injected cause and tval, step over the instruction
.align 4
s_trap:
csrr a5, scause
csrr a6, stval
csrr t0, sepc
addi t0, t0, 4
csrw sepc, t0
sret

# ================================================================ M: the monitor's frame shape
.align 12
m_entry:
CCSRRW(sp, CCSR_CSCRATCH, sp)                  # sp := the frame; S's sp parked in CSCRATCH
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
MOVC(a0, sp)                                   # a copy of the frame capability (NONLIN)
call csmint_dispatch
bnez a0, m_return
li t0, 0xbad1                                  # an illegal instruction that is not a csmint
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
# M's own capabilities do not survive an exit: the return path restores S's registers, s3 and s8 included (the
# first run, 2026-10-17, faulted 24 on m_next63's store through S's s8). Re-mint both before any use.
MINT(s3, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RW)
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
ld t0, 136(sp)                                 # S's a7
li t1, 21
beq t0, t1, m_next63
li t1, 22
beq t0, t1, m_corrupt
li t1, 99
beq t0, t1, m_final
ld t2, 96(sp)
CAPPRINT(t2)                                   # S's a2: rd
ld t2, 120(sp)
CAPPRINT(t2)                                   # S's a5: scause or -1
ld t2, 128(sp)
CAPPRINT(t2)                                   # S's a6: stval or -1
j m_skip
m_next63:
li t2, 63
sd t2, 8(s8)
j m_skip
m_corrupt:
li t2, 1
slli t2, t2, 32
sd t2, 16(s8)                                  # free count 1, free head 0
j m_skip
m_final:
lla t0, slotA
PHYS(t1, t0)
LDC(t3, t1, 0)
LCC(t2, t3, x1)
CAPPRINT(t2)                                   # 0: LINEAR, LCC's spec numbering
LCC(t2, t3, x3)
CAPPRINT(t2)                                   # dbuf + 512
LCC(t2, t3, x4)
CAPPRINT(t2)                                   # dbuf + 768
LCC(t2, t3, x2)
CAPPRINT(t2)                                   # dbuf + 512
LCC(t2, t3, x5)
CAPPRINT(t2)                                   # 4
lla t0, slotB
PHYS(t1, t0)
ld t2, 0(t1)
CAPPRINT(t2)                                   # 0
ld t2, 8(t1)
CAPPRINT(t2)                                   # 0
lla t0, slotnod
PHYS(t1, t0)
ld t2, 0(t1)
CAPPRINT(t2)                                   # 0
ld t2, 8(s8)
CAPPRINT(t2)                                   # 63
ld t2, 16(s8)
CAPPRINT(t2)                                   # 0x100000000
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
CCSRRW(sp, CCSR_CSCRATCH, sp)                  # S's sp back; the frame parked again
mret

# ================================================================ the monitor's routine, from capstone-sbi (pinned: README.md)
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
desc:
  .zero 4096
slotA:
  .zero 16
slotB:
  .zero 4080
slotnod:
  .zero 4096
dbuf:
  .zero 4096
descnoa:
  .zero 4096
frame:
  .zero 256
frame_end:
  TEST_DATA
RVTEST_DATA_END
.align 12
ram_end:
