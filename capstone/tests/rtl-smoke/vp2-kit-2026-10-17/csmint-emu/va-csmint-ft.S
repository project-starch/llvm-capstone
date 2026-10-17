# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-csmint-ft.S  (vp2: the csmint hook's FALL-THROUGH, through the monitor's own trap entry, 2026-10-17)
#-----------------------------------------------------------------------------
#
# va-csmint-entry ran the shipped _cap_trap_entry with every illegal instruction a csmint, so the hook's other branch
# never ran: csmint_dispatch returns 0 ("not ours"), the entry re-reads mcause and calls handle_exception, and the
# value it returns is written to x[rd] (rd from mtval[11:7]) with mepc advanced. On the board that is every Linux
# rdtime, which the FPGA monitor emulates in handle_exception. This arm runs that branch, then one real csmint.
#
# Same setup, pins and includes as va-csmint-entry (README.md). The stubs differ:
#   - handle_exception prints its cause argument and mtval, then returns 0x7e50 | (mtval[23:20]), standing in for the
#     monitor's time emulation (its C body runs only on vp2);
#   - handle_trap_ecall: a7 = 30 prints S's s6, s7, s9 and a3; a7 = 1 prints a2, a5, a6 as before; a7 = 99 reads
#     slotA back and passes.
#
# S's rows:
#   A `csrr s7, time` (no time CSR in the RTL: illegal, the Linux shape);
#   B `csrr s7, mhartid` (privilege);
#   C a near-miss: csmint's opcode and funct3 with funct7 0x58, one bit from 0x50 and unassigned (the decoder's
#     default arm: illegal at decode; masked 0xb000105b against csmint's 0xa000105b; 0x51 is cderive, 0x52 crevoke,
#     0x54 csnode);
#   D the real csmint of va-csmint-entry's row 1.
# Pre-registered readings: README.md.

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
li s6, 0x5106
li s9, 0x5109
li a3, 0x5a03
li s7, 0x5107                  # A: rdtime, the Linux shape
csrr s7, time
li a7, 30
ecall
li s7, 0x5107                  # B: an M-only csr from S
csrr s7, mhartid
li a7, 30
ecall
li s7, 0x5107                  # C: csmint's opcode and funct3, funct7 0x58 (unassigned)
.insn r 0x5B, 0x1, 0x58, s7, x0, x0
li a7, 30
ecall
ROW(slotA, desc, 0)            # D: va-csmint-entry's row 1
REPORT(1)
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
handle_exception:                              # the fall-through: stands in for the monitor's time emulation
  CAPPRINT(a0)                                 # the cause the entry passes: 2
  csrr t0, mtval
  CAPPRINT(t0)                                 # the instruction bits
  srli t0, t0, 20
  andi t0, t0, 15
  li a0, 0x7e50
  or a0, a0, t0                                # returned: the entry writes it to x[rd] and advances mepc
  ret
handle_trap_ecall:                             # S's registers are live here (the entry restored them from the frame)
  li t1, 30
  beq a7, t1, e_regs
  li t1, 99
  beq a7, t1, e_final
  CAPPRINT(a2)                                 # rd
  CAPPRINT(a5)                                 # scause or -1
  CAPPRINT(a6)                                 # stval or -1
  li a0, 0
  ret
e_regs:
  CAPPRINT(s6)
  CAPPRINT(s7)                                 # the value handle_exception returned, through the frame
  CAPPRINT(s9)
  CAPPRINT(a3)
  li a0, 0
  ret
e_final:
  MINT(s3, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RW)
  lla t0, slotA
  PHYS(t1, t0)
  LDC(t3, t1, 0)
  LCC(t2, t3, x1)
  CAPPRINT(t2)                                 # 0: LINEAR, lcc's spec numbering
  LCC(t2, t3, x3)
  CAPPRINT(t2)                                 # dbuf
  LCC(t2, t3, x4)
  CAPPRINT(t2)                                 # dbuf + 256
  LCC(t2, t3, x2)
  CAPPRINT(t2)                                 # dbuf
  LCC(t2, t3, x5)
  CAPPRINT(t2)                                 # 6: RW
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
