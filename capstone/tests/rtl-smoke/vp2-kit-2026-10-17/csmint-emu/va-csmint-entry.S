# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-csmint-entry.S  (vp2: the monitor's csmint path through the MONITOR'S OWN trap entry, 2026-10-17)
#-----------------------------------------------------------------------------
#
# As va-csmint-mon, with one difference: the M trap entry is not a model. This arm #includes capstone-sbi's
# sbi_capstone.S itself (which #includes capstone_csmint.S), built with CAPSTONE_TARGET_FPGA and CAPSTONE_VP2_CSMINT,
# and mtvec points at its _cap_trap_entry. So the shipped entry runs: the CSCRATCH swap, the GPR save, ldc(gp), the
# movc(a0, sp) + call save_smode_context, the ecall dispatch with its CEPC + 4, _handle_non_ecall's csmint hook, the
# rd write-back path and return_to_sumode. Pinned (README.md): the capstone-sbi commit and the sha256 of each file.
#
# What the arm supplies in place of monitor state, and therefore what only the vp2 boot proves:
#   - the four C functions the entry calls are asm stubs here: save_smode_context (returns), handle_trap_ecall (the
#     arm's row reporter, and the three table edits), handle_exception (prints 0xbad2 and the cause, then spins: no
#     csmint may fall through to it), handle_interrupt (prints 0xbad3, spins). The monitor's real C versions run only
#     on vp2: in particular handle_exception's `time` emulation after the hook, and save_smode_context's effect.
#   - the stack: CSCRATCH holds a NONLIN capability over a 4 KiB stack with its cursor at the top, the frame 32 * 8
#     below it and a capability at frame - 16 for ldc(gp). The monitor's stack capability comes from split at init
#     (sbi_capstone_init.S) and dom_init; the entry's own movc(a0, sp) followed by further use of sp, which the csmint
#     hook repeats, already runs on silicon at every boot of the current monitor.
#
# Rows and pre-registered readings: exactly va-csmint-mon's (46 values, then pass), printed by the handle_trap_ecall
# stub from S's a2, a5, a6. The entry clobbers S's a0/a1 with the SBI error/value pair on every ecall; S reloads them.

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
la t0, _cap_trap_entry
csrw mtvec, t0
csrw mcause, x0

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)
#define PHYS(REG, AREG) li t6, RAM_BASE; sub t6, AREG, t6; CINCOFFSET(REG, s3, t6)

MINT(s3, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RW)     # M's RAM capability (setup and the final read-back only)
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)        # the table
MINT(a2, CAP_TYPE_NONLIN, mstack, mstack_top, CAP_PERM_RW) # the monitor's stack: frame and gp slot below the top
li t0, 4096
CINCOFFSET(a2, a2, t0)                                      # cursor at the top, as the monitor parks it
li t0, -32 * 8 - 16
CINCOFFSET(t1, a2, t0)
STC(t1, s3, 0)                                              # a capability in the ldc(gp) slot (monitor: gp at init)
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

# ================================================================ M: the stubs for the monitor's C functions
.align 4
save_smode_context:
  ret
handle_interrupt:
  li t0, 0xbad3
  CAPPRINT(t0)
1: j 1b
handle_exception:                              # nothing in this arm may reach it: every illegal instruction is a csmint
  li t0, 0xbad2
  CAPPRINT(t0)
  CAPPRINT(a0)
1: j 1b
handle_trap_ecall:                             # S's a0..a7 are live here (the entry restored them from the frame)
  li t1, 21
  beq a7, t1, e_next63
  li t1, 22
  beq a7, t1, e_corrupt
  li t1, 99
  beq a7, t1, e_final
  CAPPRINT(a2)                                 # rd
  CAPPRINT(a5)                                 # scause or -1
  CAPPRINT(a6)                                 # stval or -1
  li a0, 0
  ret
e_next63:
  MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
  li t2, 63
  sd t2, 8(s8)
  li a0, 0
  ret
e_corrupt:
  MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
  li t2, 1
  slli t2, t2, 32
  sd t2, 16(s8)
  li a0, 0
  ret
e_final:
  MINT(s3, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RW)
  MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
  lla t0, slotA
  PHYS(t1, t0)
  LDC(t3, t1, 0)
  LCC(t2, t3, x1)
  CAPPRINT(t2)                                 # 0: LINEAR, lcc's spec numbering
  LCC(t2, t3, x3)
  CAPPRINT(t2)                                 # dbuf + 512
  LCC(t2, t3, x4)
  CAPPRINT(t2)                                 # dbuf + 768
  LCC(t2, t3, x2)
  CAPPRINT(t2)                                 # dbuf + 512
  LCC(t2, t3, x5)
  CAPPRINT(t2)                                 # 4
  lla t0, slotB
  PHYS(t1, t0)
  ld t2, 0(t1)
  CAPPRINT(t2)                                 # 0
  ld t2, 8(t1)
  CAPPRINT(t2)                                 # 0
  lla t0, slotnod
  PHYS(t1, t0)
  ld t2, 0(t1)
  CAPPRINT(t2)                                 # 0
  ld t2, 8(s8)
  CAPPRINT(t2)                                 # 63
  ld t2, 16(s8)
  CAPPRINT(t2)                                 # 0x100000000
  CAP_PASS(s11)

# ================================================================ the monitor's trap entry and routine, from capstone-sbi
# LAST in the text: sbi_capstone.h redefines ldc, stc (with the operand order swapped), movc and ccsrrw for the
# monitor's own code, so no line of the arm may follow it. (pinned: README.md)
#define CAPSTONE_TARGET_FPGA 1
#define CAPSTONE_VP2_CSMINT 1
#include "sbi_capstone.S"

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
mstack:
  .zero 4096
mstack_top:
  TEST_DATA
RVTEST_DATA_END
.align 12
ram_end:
