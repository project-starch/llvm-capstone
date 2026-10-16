# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-csmint-emu.S  (vp2 item D, the monitor's csmint emulation, 2026-10-17; the board lane's, run by the RTL lane on D4)
#-----------------------------------------------------------------------------
#
# S runs under Sv39 page tables (identity over the first 2 MiB of DRAM, 4 KiB supervisor pages) and issues csmint. On
# vp2 csmint traps to M as an illegal instruction (cause 2, mtval = the bits). M's handler is the EMULATION the monitor
# will carry (between "csmint_emulate:" and "csmint_done"), written against QEMU's helper_csmint (capstone-qemu
# target/riscv/capstone_table.c:218-283 at d35af47b163c), in QEMU's order, every refusal before any write:
#   desc & 7 -> 4 (tval desc); read {base, end, perms} with three plain loads under mstatus.MPRV (MPP = S);
#   slot & 15 -> 6 (tval slot); walk the slot VA in software (V; R, W, A and D; U only with mstatus.SUM; superpage
#   alignment) -> 15 (tval slot) and never set A or D, as the RTL's walker faults rather than updating them;
#   base >= end or perms > 15 -> 29; [base, end) not representable under the encoder (a length of 4 KiB and up needs
#   a granule-aligned base and a granule-multiple length, granule 2^(msb(len)-9): capstone_repr_ok, oracle-validated
#   by capstone/tests/exact-bounds-iii/repr-cross-check.py) -> 29;
#   mint the LINEAR capability with capcreate/captype/capbound/capperm; the table window is EXACTLY csrsi csupctl 8;
#   csnode; csrci csupctl 8 (the RTL lane's rule: under the bit the revocation-cache probes read the kernel's table);
#   csnode's own refusals (30 below two free identities, 29 corrupt) arrive as M faults inside the window, and the
#   handler clears the bit, restores mepc/mstatus and reports the cause; then stc the capability through a PHYSICAL
#   capability over the slot (s3, the RAM capability here; the monitor's region capability on the board); rd = the
#   ancestor id.
# The routine returns a status (0 = done, else the cause S must receive) and a value (the ancestor id, or the tval).
# Here the arm prints both and skips the instruction; the monitor injects the cause to S instead.
# Operand convention of this arm (checked from mtval): rd = a2, rs1 = a0 (the slot VA), rs2 = a1 (the descriptor VA).
#
# Rows, in order (the table: RECYCLE, capacity 64, next-unused 3, record 2 valid, as va-runv6):
#   1 good: D1 {dbuf, dbuf+256, RW} into slotA          status 0, value 4 (child 3, ancestor 4)
#     then the table (ecall 1): next-unused 5; rec3 w1 0x3000000100000001; rec4 w1 0x3000000000000000
#   2 descriptor misaligned (D1+4)                       status 4, value D1+4
#   3 slot misaligned (slotB+8)                          status 6, value slotB+8
#   4 base >= end: D2 {dbuf+256, dbuf, RW}               status 29, value 0
#   5 unrepresentable: D3 {dbuf+8, dbuf+8+8192, RW}      status 29, value 0  (8192: granule 16, base % 16 = 8)
#   6 perms > 15: D5 {dbuf, dbuf+256, 16}                status 29, value 0
#   7 slot PTE without D (slotnod's page: V R W A)       status 15, value slotnod  (nothing written)
#   8 second csmint on slotA: D4 {dbuf+512, dbuf+768, RO}  status 0, value 6 (child 5, ancestor 6): QEMU overwrites
#     then the table (ecall 8): next-unused 7; rec5 w1 0x3000000180000001; rec6 w1 0x3000000000000000
#   9 the reserve: next-unused set to 63 (one free identity, ecall 9), D1 into slotB   status 30 (csnode's fault)
#   final (ecall 99): slotA read back through the physical capability: type 1, base dbuf+512, end dbuf+768,
#     cursor dbuf+512, perms 4; slotB, slotB+8 and slotnod's first word still 0; then pass.
# Record words as va-runv6: w1 = {free[62], linear[61], valid[60], parent[45:30], depth[16:0]}.
# QEMU DIFFERS ON ROW 7 BY CONFIGURATION, NOT BY DESIGN: its svadu extension is on by default (target/riscv/cpu.c:1919)
# and its walker then SETS D (cpu_helper.c:1123-1130), so QEMU's own csmint succeeds there; the RTL walker faults on
# a clear A/D bit and the emulation follows the RTL. On silicon Linux sets D in its page-fault handler and the retried
# csmint succeeds, so the end state agrees; the count of faults does not.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSNODE(rd, rs1)       .insn r 0x5B, 0x1, 0x54, rd, rs1, x0
#define CSMINT(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x50, rd, rs1, rs2

#define SREVROOT 0x5C1
#define CSUPCTL  0x7C4

#define PTE_S_RWX   0xCF   /* V R W X A D, supervisor */
#define PTE_S_NOD   0x47   /* V R W A: no D */
#define PTE_PTR     0x01

#define RAM_BASE 0x80000000

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

# a capability in REG at the physical address in AREG, through s3 (the RAM capability, base RAM_BASE)
#define PHYS(REG, AREG) li t6, RAM_BASE; sub t6, AREG, t6; CINCOFFSET(REG, s3, t6)
#define PREC1(I) li t0, 16*(I); CINCOFFSET(t1, s8, t0); ld t2, 8(t1); CAPPRINT(t2)

# ---------------------------------------------------------------- M's capabilities
MINT(s3, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RW)     # the RAM capability: the walk and the physical stc
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)        # the table

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

# ---------------------------------------------------------------- the descriptors {base, end, perms}
#define DESC(OFF, B, E, P) lla t0, desc; addi t0, t0, OFF; PHYS(t1, t0); \
  lla t2, dbuf; li t3, B; add t3, t3, t2; sd t3, 0(t1); li t3, E; add t3, t3, t2; sd t3, 8(t1); li t3, P; sd t3, 16(t1)
DESC(0,   0,   256,      CAP_PERM_RW)     # D1
DESC(32,  256, 0,        CAP_PERM_RW)     # D2: base >= end
DESC(64,  8,   8200,     CAP_PERM_RW)     # D3: [dbuf+8, dbuf+8+8192): unrepresentable
DESC(96,  512, 768,      CAP_PERM_RO)     # D4
DESC(128, 0,   256,      16)              # D5: perms > 15

# ---------------------------------------------------------------- Sv39 page tables (physical, built bare)
# L0: 512 x 4 KiB identity supervisor leaves over [RAM_BASE, RAM_BASE + 2 MiB)
lla t0, pt_l0
PHYS(s4, t0)                   # s4: L0
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
# L0[slotnod] -> V R W A, no D
lla t1, slotnod
li t2, RAM_BASE
sub t2, t1, t2
srli t2, t2, 12
slli t2, t2, 3
CINCOFFSET(t0, s4, t2)
srli t3, t1, 12
slli t3, t3, 10
ori t3, t3, PTE_S_NOD
sd t3, 0(t0)
# L1: [0] -> L0
lla t0, pt_l1
PHYS(t1, t0)
lla t2, pt_l0
srli t2, t2, 12
slli t2, t2, 10
ori t2, t2, PTE_PTR
sd t2, 0(t1)
# root[2] -> L1 (va[38:30] = 2 for 0x80000000)
lla t0, pt_root
PHYS(t1, t0)
lla t2, pt_l1
srli t2, t2, 12
slli t2, t2, 10
ori t2, t2, PTE_PTR
sd t2, 16(t1)
# PMP for the walker's S-mode reads, and the CPMP for S's fetches
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
MINT(a1, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RWX)
CCSRRW(x0, CCSR_CPMP(0), a1)
# satp = SV39 | root ppn
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma

li s9, 0                       # s9: 1 only while the csupctl.table window is open

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
#define ROW(SLOT, DESC_OFF) lla a0, SLOT; lla a1, desc; addi a1, a1, DESC_OFF; CSMINT(a2, a0, a1)
ROW(slotA, 0)                  # 1: status 0, value 4
li a7, 1
ecall
lla a0, slotA                  # 2: descriptor misaligned
lla a1, desc
addi a1, a1, 4
CSMINT(a2, a0, a1)
lla a0, slotB                  # 3: slot misaligned
addi a0, a0, 8
lla a1, desc
CSMINT(a2, a0, a1)
ROW(slotB, 32)                 # 4: base >= end, 29
ROW(slotB, 64)                 # 5: unrepresentable, 29
ROW(slotB, 128)                # 6: perms > 15, 29
ROW(slotnod, 0)                # 7: no D, 15
ROW(slotA, 96)                 # 8: second csmint on slotA, status 0, value 6
li a7, 8
ecall
li a7, 9                       # one free identity
ecall
ROW(slotB, 0)                  # 9: 30
li a7, 99
ecall
1: j 1b

# ================================================================ M
.align 12
trap_handler:
  bnez s9, window_fault        # csnode refused inside the window (30 or 29)
  csrr t0, mcause
  li t1, 2
  beq t0, t1, maybe_csmint
  li t1, 9
  beq t0, t1, s_ecall
  CAPPRINT(t0)                 # unexpected: print the cause and skip
  j skip
maybe_csmint:
  csrr t0, mtval
  li t1, 0xfe00707f
  and t2, t0, t1
  li t1, 0xa000105b
  bne t2, t1, not_csmint
  srli t2, t0, 7               # this arm's operand convention: rd a2, rs1 a0, rs2 a1
  andi t2, t2, 31
  li t1, 12
  bne t2, t1, bad_regs
  srli t2, t0, 15
  andi t2, t2, 31
  li t1, 10
  bne t2, t1, bad_regs
  srli t2, t0, 20
  andi t2, t2, 31
  li t1, 11
  bne t2, t1, bad_regs
  mv a3, a1                    # the descriptor VA
  mv a4, a0                    # the slot VA
  j csmint_emulate
not_csmint:
  li t0, 0xbad1
  CAPPRINT(t0)
  j skip
bad_regs:
  li t0, 0xbad2
  CAPPRINT(t0)
  j skip

# ---------------------------------------------------------------- the emulation (in: a3 desc VA, a4 slot VA; out: a5 status, a6 value)
csmint_emulate:
  andi t0, a3, 7               # desc & 7 -> 4
  beqz t0, 1f
  li a5, 4
  mv a6, a3
  j csmint_done
1:
  li t0, 0x20000               # mstatus.MPRV: the three words through S's translation (MPP = S from the trap)
  csrs mstatus, t0
  ld t3, 0(a3)                 # base
  ld t4, 8(a3)                 # end
  ld t5, 16(a3)                # perms
  csrc mstatus, t0
  andi t0, a4, 15              # slot & 15 -> 6
  beqz t0, 2f
  li a5, 6
  mv a6, a4
  j csmint_done
2:
  # ---- the software walk of the slot VA (a4): Sv39, physical tables read through s3
  csrr t0, satp
  slli t0, t0, 20              # ppn[43:0]
  srli t0, t0, 20
  slli t0, t0, 12              # root PA
  li a7, 2                     # level
3:
  li t1, 9
  mul t1, t1, a7
  addi t1, t1, 12              # shift = 12 + 9 * level
  srl t2, a4, t1
  andi t2, t2, 511
  slli t2, t2, 3
  add t2, t2, t0               # the PTE's PA
  PHYS(t6, t2)
  ld t6, 0(t6)                 # the PTE
  andi t2, t6, 1               # V
  beqz t2, walk_fault
  andi t2, t6, 0xA             # R or X: a leaf
  bnez t2, 4f
  srli t0, t6, 10              # a pointer: next table = ppn << 12
  slli t0, t0, 20
  srli t0, t0, 20
  slli t0, t0, 12
  addi a7, a7, -1
  bltz a7, walk_fault
  j 3b
4:
  andi t2, t6, 0xC6            # the store needs R, W, A and D; never set A or D on S's behalf
  li a6, 0xC6
  bne t2, a6, walk_fault
  andi t2, t6, 0x10            # U: only with mstatus.SUM
  beqz t2, 5f
  csrr t2, mstatus
  srli t2, t2, 18
  andi t2, t2, 1
  beqz t2, walk_fault
5:
  li a6, 1                     # mask = (1 << shift) - 1
  sll a6, a6, t1
  addi a6, a6, -1
  srli t2, t6, 10
  slli t2, t2, 20
  srli t2, t2, 20
  slli t2, t2, 12              # the leaf's PA
  and t0, t2, a6               # a misaligned superpage -> fault
  bnez t0, walk_fault
  and t0, a4, a6
  or a4, t2, t0                # a4: the slot's PA from here on
  # ---- the descriptor: base < end, perms <= 15, representable
  bgeu t3, t4, bad_desc
  li t0, 15
  bltu t0, t5, bad_desc
  sub t0, t4, t3               # len
  li t1, 4096
  bltu t0, t1, 7f              # below 4 KiB: exact
  li t1, 63                    # msb(len): the highest set bit
  li t2, 1
6:
  sll t6, t2, t1
  and t6, t6, t0
  bnez t6, 61f
  addi t1, t1, -1
  j 6b
61:
  addi t1, t1, -9
  sll t1, t2, t1               # granule = 2^(msb - 9)
  addi t1, t1, -1
  and t6, t3, t1               # base % granule
  bnez t6, bad_desc
  and t6, t0, t1               # len % granule
  bnez t6, bad_desc
7:
  # ---- mint, csnode in the window, stc through the physical capability
  CAPCREATE(s5)
  li t0, CAP_TYPE_LIN
  CAPTYPE(s5, t0)
  CAPBOUND(s5, t3, t4)
  CAPPERM(s5, t5)
  csrr s10, mepc               # a csnode refusal traps from M and overwrites these
  csrr s7, mstatus
  li s9, 1
  csrsi CSUPCTL, 8
  CSNODE(a6, s5)
  csrci CSUPCTL, 8
  li s9, 0
  PHYS(t0, a4)
  STC(t0, s5, 0)
  li a5, 0                     # status 0, a6 = the ancestor
  j csmint_done
walk_fault:
  li a5, 15
  mv a6, a4
  j csmint_done
bad_desc:
  li a5, 29
  li a6, 0
  j csmint_done
window_fault:
  csrci CSUPCTL, 8
  li s9, 0
  csrr a5, mcause
  li a6, 0
  csrw mepc, s10
  csrw mstatus, s7
csmint_done:
  CAPPRINT(a5)
  CAPPRINT(a6)
  bnez a5, skip
  mv a2, a6                    # rd
skip:
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

# ---------------------------------------------------------------- S's ecalls
s_ecall:
  li t1, 1
  beq a7, t1, e_table1
  li t1, 8
  beq a7, t1, e_table8
  li t1, 9
  beq a7, t1, e_reserve
  li t1, 99
  beq a7, t1, e_final
  li t0, 0xbad3
  CAPPRINT(t0)
  j skip
e_table1:
  ld t2, 8(s8)
  CAPPRINT(t2)                 # 5
  PREC1(3)                     # 0x3000000100000001
  PREC1(4)                     # 0x3000000000000000
  j skip
e_table8:
  ld t2, 8(s8)
  CAPPRINT(t2)                 # 7
  PREC1(5)                     # 0x3000000180000001
  PREC1(6)                     # 0x3000000000000000
  j skip
e_reserve:
  li t0, 63
  sd t0, 8(s8)
  j skip
e_final:
  lla t0, slotA
  PHYS(t1, t0)
  LDC(s5, t1, 0)
  LCC(t2, s5, x1)
  CAPPRINT(t2)                 # 1: LINEAR
  LCC(t2, s5, x3)
  CAPPRINT(t2)                 # base: dbuf + 512
  LCC(t2, s5, x4)
  CAPPRINT(t2)                 # end: dbuf + 768
  LCC(t2, s5, x2)
  CAPPRINT(t2)                 # cursor: dbuf + 512
  LCC(t2, s5, x5)
  CAPPRINT(t2)                 # perms: 4
  lla t0, slotB
  PHYS(t1, t0)
  ld t2, 0(t1)
  CAPPRINT(t2)                 # 0
  ld t2, 8(t1)
  CAPPRINT(t2)                 # 0
  lla t0, slotnod
  PHYS(t1, t0)
  ld t2, 0(t1)
  CAPPRINT(t2)                 # 0
  CAP_PASS(s11)

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
  TEST_DATA
RVTEST_DATA_END
.align 12
ram_end:
