# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-osinject.S  (vp2: the monitor injects an OS-origin trap into S, through the MONITOR'S OWN trap entry, 2026-10-17)
#-----------------------------------------------------------------------------
#
# vp2 Tier 1 boot 1 wedged on a cause-24 trap from S (a kernel module's ldc on an integer base). The monitor tore
# down a domain that was not there, never ran return_to_sumode, and the next trap's cincoffsetimm(sp) at
# _cap_trap_entry+4 faulted on S's integer sp forever. capstone-sbi vp2/os-trap-inject (db23ffd) makes
# handle_exception return CAPSTONE_INJECT_S (-2) for a trap from S or U, and the entry's _inject_s_trap delivers it
# into S (csm_inject's sequence, SPP from MPP), leaving through return_to_sumode.
#
# This arm #includes that sbi_capstone.S, built with CAPSTONE_TARGET_FPGA and CAPSTONE_VP2_CSMINT, as va-csmint-entry
# does, with mtvec at its _cap_trap_entry. So the shipped entry runs: the CSCRATCH swap, the GPR save, the -2 branch,
# _inject_s_trap, _handle_branch_1 and return_to_sumode.
#
# The C functions the entry calls are asm stubs, as in va-csmint-entry. handle_exception's stub is the patched C's
# decision for this arm's traps: MPP S or U -> return -2. The C itself (three sentinel sites, the MPP test, the
# dom_calls_active guard in fault_return_from_domain, eight bracketed domain calls) is checked in the generated
# monitor of the firmware built from db23ffd, not here. -DOSINJECT_CONTROL makes the stub return -1 instead (resume
# without injecting): the ldc re-traps forever and no row is printed.
#
# Rows (CAPPRINT, then pass), printed by the handle_trap_ecall stub from S's registers:
#   row 1, S origin: scause (a5), stval (a6), sstatus.SPP (a3), sepc (a4), t2, s5, s6
#   row 2, U origin: the same seven
# Row 1's own ecall is the SECOND trap after an injection: it prints only if the entry swaps CSCRATCH back.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define PTE_S_RWX   0xCF   /* V R W X A D, supervisor */
#define PTE_U_RWX   0xDF   /* V R W X U A D */
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

MINT(s3, CAP_TYPE_NONLIN, _start, ram_end, CAP_PERM_RW)     # M's RAM capability (setup only)
MINT(a2, CAP_TYPE_NONLIN, mstack, mstack_top, CAP_PERM_RW) # the monitor's stack: frame and gp slot below the top
li t0, 4096
CINCOFFSET(a2, a2, t0)                                      # cursor at the top, as the monitor parks it
li t0, -32 * 8 - 16
CINCOFFSET(t1, a2, t0)
STC(t1, s3, 0)                                              # a capability in the ldc(gp) slot
CCSRRW(x0, CCSR_CSCRATCH, a2)

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
L0_SET(u_code, PTE_U_RWX)                                   # U fetches its one page; its ldc faults before memory
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
li s5, 0x5555                                  # callee-saved: preserved by the entry's frame
li s6, 0x6666
#define PRE li a3, -1; li a4, -1; li a5, -1; li a6, -1; li t2, 0x7777
#define REPORT(K) li a7, K; ecall

# row 1: an ldc from S on an INTEGER base: cause 24, which medeleg cannot route (Q-24), injected by the monitor
PRE
lla a0, slotA
ldc_s:
LDC(t0, a0, 0)
REPORT(1)                                      # the second trap after the injection

# row 2: the same from U
PRE
lla t0, u_code
csrw sepc, t0
li t0, 0x100
csrc sstatus, t0                               # SPP = 0: sret to U
sret
s_after_u:
REPORT(2)
REPORT(99)

# S's own trap handler: record the injected trap; from S step over the ldc, from U come back to S at s_after_u
.align 4
s_trap:
csrr a5, scause
csrr a6, stval
csrr a3, sstatus
csrr a4, sepc
srli a3, a3, 8
andi a3, a3, 1                                 # SPP
bnez a3, 1f
la t0, s_after_u
csrw sepc, t0
li t0, 0x100
csrs sstatus, t0                               # back to S
sret
1:
csrr t0, sepc
addi t0, t0, 4
csrw sepc, t0
sret

# ================================================================ U: one page
.align 12
u_code:
lla a0, slotA
ldc_u:
LDC(t0, a0, 0)
1: j 1b                                        # not reached: s_trap returns to S

# ================================================================ M: the stubs for the monitor's C functions
.align 12
save_smode_context:
  ret
handle_interrupt:
  li t0, 0xbad3
  CAPPRINT(t0)
1: j 1b
handle_exception:                              # the patched C's decision for this arm's traps
  csrr t0, mstatus
  srli t0, t0, 11
  andi t0, t0, 3
  li t1, 3
  beq t0, t1, 2f
#ifdef OSINJECT_CONTROL
  li a0, -1                                    # control: resume without injecting
#else
  li a0, -2                                    # CAPSTONE_INJECT_S
#endif
  ret
2:
  li t0, 0xbad2                                # no trap from M is expected in this arm
  CAPPRINT(t0)
  CAPPRINT(a0)
1: j 1b
handle_trap_ecall:                             # S's a0..a7 are live here (the entry restored them from the frame)
  li t1, 99
  beq a7, t1, e_final
  CAPPRINT(a5)                                 # scause
  CAPPRINT(a6)                                 # stval
  CAPPRINT(a3)                                 # sstatus.SPP
  CAPPRINT(a4)                                 # sepc
  CAPPRINT(t2)                                 # 0x7777: a register _inject_s_trap clobbers, restored by the exit
  CAPPRINT(s5)                                 # 0x5555
  CAPPRINT(s6)                                 # 0x6666
  li a0, 0
  ret
e_final:
  CAP_PASS(s11)

# ================================================================ the monitor's trap entry, from capstone-sbi
# LAST in the text: sbi_capstone.h redefines ldc, stc (operand order swapped), movc and ccsrrw for the monitor's own
# code, so no line of the arm may follow it. (pinned: README.md)
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
slotA:
  .zero 4096
mstack:
  .zero 4096
mstack_top:
  TEST_DATA
RVTEST_DATA_END
.align 12
ram_end:
