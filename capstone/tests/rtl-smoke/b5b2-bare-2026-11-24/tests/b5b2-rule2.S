# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-rule2.S -- B5b-2 step 3: rule 2 in the MMU (a plain S access reaches no user page while srevroot != 0) (2026-11-24)
#-----------------------------------------------------------------------------
#
# The reference (capstone-qemu 9289114e06 riscv_cpu_mmu_index): while the HART is in S and srevroot is nonzero, a plain
# access ignores SUM, so a user page faults (13 load, 15 store), and an access through a reference translates with S's
# SUM index, reaching user pages whatever SUM says (rule 1). M with MPRV = 1 and MPP = S keeps SUM.
# Table and capabilities: b5b2-uprot.S's (x26/s10 = {0,3} live over data, x27/s11 = {0,4} stale, x24/s8 data as an
# integer). Translation: sublet-ns.S's Sv39 identity map of 0x8000_0000.., every page a USER page (PTE U set) except
# s_code's, which S runs from. M runs bare. Traps go to M (medeleg 0); the handler records mcause in x9/s1, counts the
# trap in x28/t3, resumes at mepc + 4; an ecall returns to M's continuation at x30/t5. data[0] = 0x77; stores go to
# data + 16.
#   phase A, S, srevroot = tbl, SUM = 1:  a0 cause of ld x24 (plain)   a1 cause of sd x24+16 (plain)
#                                         a2 cause of ld x26 (a reference)   a3 its value
#   phase B, S, srevroot = tbl, SUM = 0:  a4 cause of ld x26   a5 its value   a6 cause of ld x24 (plain)
#   phase C, S, srevroot = 0,   SUM = 1:  a0 a1 cause and value of ld x24   a2 a3 cause and value of ld x27
#   phase D, M, MPRV = 1, MPP = S, srevroot = tbl, SUM = 1:  a0 a1 cause and value of ld x24
#   then the trap count (ecalls not counted)
# PRE-REGISTERED (2026-11-24, written before any run), Sublet build:
#   step 3 and the reference:  A 0xd 0xf 0 0x77 | B 0 0x77 0xd | C 0 0x77 0 0x77 | D 0 0x77 | traps 3
#   step 2 (b5b2-step2 94cbc880e, the base):  A 0 0 0 0x77 | B 0xd 0 0xd | C 0 0x77 0 0x77 | D 0 0x77 | traps 2
#   (B's a4 on the base: x26 passes step 2's capability check, then the MMU applies SUM = 0 to a user page.)
# FIRST RUN (b3329c988): every value as pre-registered on both trees except the trap count, which read 0x2ff800e1 /
#   0x2ff800e2: the page-table block (sublet-ns.S's) uses t3 = x28, the counter, as a temporary. Its difference (+1)
#   was the predicted one. The counter is now cleared after the page tables; nothing else changed.
# -DR2_PAIR (the step-3 claim-auditor's leak test, added before any run of it): phase E, S, srevroot = tbl, SUM = 0, after
#   phase C. Each pair is two independent accesses back to back, each pair after an sfence.vma (a cold DTLB):
#     E1 ld x26 then ld x24 (a reference, then plain to a user page)    -> a0 the pair's trap count, a1 its last cause
#     E2 ld x29 then ld x26 (plain to S's own page, then a reference)   -> a2 trap count, a3 last cause, a4 x26's value
#     E3 sd x26+16 then sd x24+16                                       -> a5 trap count, a6 last cause
#   (nothing runs between the two accesses of a pair; the trap count is x28/t3's difference across it)
#   x29/t4 = s_code as an integer (a page without U). PRE-REGISTERED: step 3 E1 1 0xd | E2 0 0 0x77 | E3 1 0xf, final
#   trap count 3 + 2 = 5; the base E1 2 0xd | E2 1 0xd 0 | E3 2 0xf, final count 2 + 5 = 7. A wrong answer here is the registered
#   bit belonging to the NEIGHBOURING access. A right one does not prove the two were judged in adjacent cycles
#   (instruction adjacency is not cycle adjacency).
# FIRST RUN of -DR2_PAIR (b60a63efc): phase E exactly as pre-registered on both trees; phase D read 0xd 0 on both,
#   because D inherited SUM and phase E had cleared it (M with MPRV and SUM = 0: 13 is right). D now sets SUM itself.
# B5b-2 (2026-11-24, the semantics owner's ruling): uses the capcreate family, to be ported to CSMINT.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT    0x5C1
#define PTE_LEAF_RWX  0xDF
#define PTE_LEAF_SRWX 0xCF
#define PTE_PTR       0x01
#define SUM_BIT 0x40000
#define MPRV_BIT 0x20000

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

#define CLR  li x9, 0
#define PRINT8 CAPPRINT(a0); CAPPRINT(a1); CAPPRINT(a2); CAPPRINT(a3); CAPPRINT(a4); CAPPRINT(a5); CAPPRINT(a6); CAPPRINT(a7)

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
MINT(x18, CAP_TYPE_NONLIN, _start, pt_end, CAP_PERM_RWX)  # pool node 2: {0,2}
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
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
li x28, 0                      # the trap count: the page-table block above uses t3 (= x28) as a temporary

# ---------------------------------------------------------------- phase A: S, srevroot = tbl, SUM = 1
lla t0, tbl
csrw SREVROOT, t0
li t0, SUM_BIT
csrs mstatus, t0
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0               # MPP = S
lla t0, s_phaseA
csrw mepc, t0
lla x30, m_afterA
mret
m_afterA:
# ---------------------------------------------------------------- phase B: S, srevroot = tbl, SUM = 0
li t0, SUM_BIT
csrc mstatus, t0
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0
lla t0, s_phaseB
csrw mepc, t0
lla x30, m_afterB
mret
m_afterB:
CAPPRINT(a0); CAPPRINT(a1); CAPPRINT(a2); CAPPRINT(a3); CAPPRINT(a4); CAPPRINT(a5); CAPPRINT(a6)
# ---------------------------------------------------------------- phase C: S, srevroot = 0, SUM = 1
csrw SREVROOT, x0
li t0, SUM_BIT
csrs mstatus, t0
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0
lla t0, s_phaseC
csrw mepc, t0
lla x30, m_afterC
mret
m_afterC:
CAPPRINT(a0); CAPPRINT(a1); CAPPRINT(a2); CAPPRINT(a3)
#ifdef R2_PAIR
# ---------------------------------------------------------------- phase E: S, srevroot = tbl, SUM = 0, the pairs
lla t0, tbl
csrw SREVROOT, t0
li t0, SUM_BIT
csrc mstatus, t0
lla x29, s_code
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0
lla t0, s_phaseE
csrw mepc, t0
lla x30, m_afterE
mret
m_afterE:
CAPPRINT(a0); CAPPRINT(a1); CAPPRINT(a2); CAPPRINT(a3); CAPPRINT(a4); CAPPRINT(a5); CAPPRINT(a6)
#endif
# ---------------------------------------------------------------- phase D: M, MPRV = 1, MPP = S, srevroot = tbl, SUM = 1
lla t0, tbl
csrw SREVROOT, t0
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0               # MPP = S
li t0, SUM_BIT
csrs mstatus, t0               # SUM = 1, set here (phase E clears it)
li t0, MPRV_BIT
csrs mstatus, t0
CLR; li t1, 0
ld t1, 0(x24)
mv a0, x9
mv a1, t1
li t0, MPRV_BIT
csrc mstatus, t0
CAPPRINT(a0); CAPPRINT(a1)
csrw SREVROOT, x0
CAPPRINT(x28)
RVTEST_PASS

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

# ---------------------------------------------------------------- the S code: one page, mapped without U
.align 12
s_code:
s_phaseA:
  CLR; li t1, 0; ld t1, 0(x24);    mv a0, x9
  CLR; li t2, 0x55; sd t2, 16(x24); mv a1, x9
  CLR; li t1, 0; ld t1, 0(x26);    mv a2, x9; mv a3, t1
  ecall
s_phaseB:
  CLR; li t1, 0; ld t1, 0(x26);    mv a4, x9; mv a5, t1
  CLR; li t1, 0; ld t1, 0(x24);    mv a6, x9
  ecall
s_phaseC:
  CLR; li t1, 0; ld t1, 0(x24);    mv a0, x9; mv a1, t1
  CLR; li t1, 0; ld t1, 0(x27);    mv a2, x9; mv a3, t1
  ecall
#ifdef R2_PAIR
s_phaseE:
  li x9, 0; sfence.vma; li t1, 0; li t2, 0; mv a0, x28
  ld t1, 0(x26)
  ld t2, 0(x24)
  sub a0, x28, a0; mv a1, x9
  li x9, 0; sfence.vma; li t1, 0; li t2, 0; mv a2, x28
  ld t1, 0(x29)
  ld t2, 0(x26)
  sub a2, x28, a2; mv a3, x9; mv a4, t2
  li x9, 0; sfence.vma; li t2, 0x55; mv a5, x28
  sd t2, 16(x26)
  sd t2, 16(x24)
  sub a5, x28, a5; mv a6, x9
  ecall
#endif
.align 12
s_code_end:
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
