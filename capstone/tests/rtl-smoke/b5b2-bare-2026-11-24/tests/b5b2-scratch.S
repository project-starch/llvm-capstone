# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-scratch.S -- B5b-2 step 4: sscratch and mscratch hold a reference with its tag
#-----------------------------------------------------------------------------
#
# The reference: capstone-qemu 9289114e06 with x-capstone-trap-route (op_helper.c helper_capstone_scratch_swap and
# helper_capstone_scratch_rmw; trans_rvi.c.inc). csrrw moves the whole register both ways: the CSR takes rs1 with its
# tag and rd receives the old value with its tag, also when rd = rs1. csrr reads the address, untagged, and the CSR
# keeps the tag. Every other write clears the tag: csrrs / csrrc with a source register (even one holding 0), csrrwi,
# csrrsi / csrrci with a nonzero immediate, csrw of an integer. csrrs / csrrc with x0 and csrrsi / csrrci with 0 write
# nothing and keep it. U cannot touch sscratch or srevroot (2), and scapctl (0x5C0) does not exist (2). The cases are
# tests/trap-route/route.S's sscratch_keeps_tag (3), mscratch_keeps_tag (4) and csr_access (2), on this file's own
# table and capabilities.
# Table and capabilities: b5b2-sref.S's setup (bare translation). x26/s10 = {0,3} live over data, x20/s4 = x26 moved
# to data + 4088 (a load at 8(x20) is one past the bounds), x24/s8 data as an integer, x29/t4 refbox as an integer,
# a6/x16 = data + 4088 as an integer. Traps go to M; the handler records mcause in x9/s1, counts in x28/t3 (zeroed after
# the setup), resumes at mepc + 4; ecall returns to M at x30/t5. A reading is stored at x25/s9 (S records too) and M
# prints them all at the end, after their count. Type is LCC selector 1: 1 = a tagged NONLIN, 7 = untagged.
# Readings, in order (S with srevroot = tbl unless stated):
#   S1   1 type of t1 after csrw sscratch, x26 and csrrw t1 back   2 cause of a cderive from t1   3 the child's type
#        4 csrr sscratch == 0 (the swap left the integer 0)          5 that csrr's type
#   S2   6 type of x22 after csrrw x22, sscratch, x22 (x22 = x20, the CSR = x26)   7 x22 == data
#        8 type of what then comes back (x20)   9 its value == data + 4088   10 cause of ld 8(it)
#   S3   11 csrr == data   12 its type   13 type of the swap back (csrr kept the tag)
#   S4   14 type after csrrs zero, sscratch, x24   15 its value == data
#   S5   16 type of csrrci t2, sscratch, 0   17 type of the swap back (no write)
#   S6   18 type after csrrwi zero, sscratch, 5   19 its value == 5
#   S7   20 type after csrw sscratch, x24 (integer)
#   S8   21 type of csrrs x22, sscratch, t2 (t2 = 0)   22 type of the swap back   23 its value == data
#   S9   24 type after csrrc zero, sscratch, t2 (t2 = 0)   25 its value == data
#   S10  26 type after csrrs x22, sscratch, zero and csrrsi x22, sscratch, 0
#   S11  27 type after csrrci zero, sscratch, 1   28 its value == data
#   S12  29 cause of csrw 0x5c0 in S   30 csrr srevroot == tbl (the positive control: S reaches 0x5C1)
#   M    31 type of csrrw t1, sscratch, zero in M (S parked x20 there before its ecall)   32 its value == data + 4088
#   U    (protected: srevroot = tbl; M parks x20 in sscratch, x22 = 0) U runs csrrw x22, sscratch, x24 and
#        csrw srevroot, zero. Then M: 33 x22 == 0   34 type of the swap back   35 its value == data + 4088
#        36 srevroot == tbl   37 traps in U
#   M    38 type of csrrw t2, mscratch, zero after csrw mscratch, x26   39 csrr mscratch == 0   40 csrw mscratch, t2,
#        then csrr == data   41 its type   42 type after csrrs zero, mscratch, x24   43 type after csrw mscratch, t2 and
#        csrrc zero, mscratch, x23 (x23 = 0)   44 type after csrw mscratch, t2 and csrrs zero, mscratch, zero
#        45 type of x22 after csrw mscratch, x26 and csrrw x22, mscratch, x22 (x22 = x20)   46 x22 == data
#        47 type of csrrw x23, mscratch, zero   48 x23 == data + 4088   (M stores x23 at refbox with stc)
#        49 cause of csrr 0x5c0 in M
#   S    50 cause of ldc x22 <- 0(x29)   51 its type   52 cause of ld 8(x22)
#   53 the trap count
# PRE-REGISTERED (2026-11-24, written before any run; derived from the reference's helpers and route.S's checks 1-26,
# 30-37, 8-9):
#   step 4 (b5b2-step4):
#     1 0 1 1 7 | 1 1 1 1 0x1c | 1 7 1 | 7 1 | 7 1 | 7 1 | 7 | 7 7 1 | 7 1 | 1 | 7 1 | 2 1 | 1 1 |
#     1 1 1 1 2 | 1 1 1 7 7 7 1 1 1 1 1 2 | 0 1 0x1c | 6
#   base (b5b2-step2b b2f27b3b3, no tag in either CSR): every type 7, every value flag 1, and
#     1: 7, 2: 0x18 (a cderive from an integer parent; predicted from the reference's "an integer parent raises 24",
#     not from the RTL), 3: 7, 10: 0 (a plain load: the base is untagged), 50: 0, 52: 0, causes 29 / 49: 2,
#     37: 2, 53: 5 (S1's cderive, S12, U's two, M's 0x5c0)
#   i.e. 7 0x18 7 1 7 | 7 1 7 1 0 | 1 7 7 | 7 1 | 7 7 | 7 1 | 7 | 7 7 1 | 7 1 | 7 | 7 1 | 2 1 | 7 1 |
#     1 7 1 1 2 | 7 1 1 7 7 7 7 7 1 7 1 2 | 0 7 0 | 5
# FIRST RUN (7c8550cf8, test as committed there): VOID. Both trees read the base vector above value for value, at the
#   same 39,529 cycles: the metadata file's read port 0 reads rs1 only for capability ops and loads/stores
#   (issue_read_operands cap_raddr_pack), so a csrrw carried x0's lane. Fixed in the RTL. The same run exposed a
#   defect here: S2 and M6 copied x20 with mv, an integer op, so those arms would not have moved a capability; they
#   copy with cincoffset by x0 now. No prediction changes: the vectors above assumed a tagged copy.
# Not covered here: a CSRUNV switcher write of sscratch (id 65) clears the tag by design (the switcher moves sscratch as an
# integer); contexts run at M, so a tagged sscratch does not survive a context's save and restore.
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
lla x25, res                   # the reading pointer (x25/s9): S records too, M prints them all
li x28, 0                      # the trap count, after every setup block

#define REC       sd a5, 0(x25); addi x25, x25, 8
#define TY(R)     li a5, 0; LCC(a5, R, x1); REC
#define EQ(R, V)  sub a5, R, V; seqz a5, a5; REC
#define EQI(R, I) li a5, I; sub a5, R, a5; seqz a5, a5; REC
#define CAUSE     mv a5, x9; REC
#define TO_S(LBL, BACK) li t0, 0x1800; csrc mstatus, t0; li t0, 0x800; csrs mstatus, t0; lla t0, LBL; csrw mepc, t0; lla x30, BACK; mret
#define TO_U(LBL, BACK) li t0, 0x1800; csrc mstatus, t0; lla t0, LBL; csrw mepc, t0; lla x30, BACK; mret

li a6, 4088
add a6, a6, x24                # data + 4088, an integer
lla t0, tbl
csrw SREVROOT, t0

# ---------------------------------------------------------------- S: sscratch
TO_S(s_scr, m_after_s)
m_after_s:
csrrw t1, sscratch, zero
TY(t1); EQ(t1, a6)

# ---------------------------------------------------------------- U: csr_access (protected U, srevroot = tbl)
csrw sscratch, x20
li x22, 0
mv x31, x28
TO_U(u_scr, m_after_u)
m_after_u:
EQI(x22, 0)
csrrw t1, sscratch, zero
TY(t1); EQ(t1, a6)
csrr t2, SREVROOT
lla t1, tbl
EQ(t2, t1)
sub a5, x28, x31; REC

# ---------------------------------------------------------------- M: mscratch
csrw mscratch, x26
li t2, 0
csrrw t2, mscratch, zero
TY(t2)
csrr x22, mscratch
EQI(x22, 0)
csrw mscratch, t2
csrr x22, mscratch
EQ(x22, x24); TY(x22)
csrrs zero, mscratch, x24
csrrw x22, mscratch, zero
TY(x22)
li x23, 0
csrw mscratch, t2
csrrc zero, mscratch, x23
csrrw x22, mscratch, zero
TY(x22)
csrw mscratch, t2
csrrs zero, mscratch, zero
csrrw x22, mscratch, zero
TY(x22)
csrw mscratch, x26
CINCOFFSET(x22, x20, x0)       # a tagged copy of x20 (mv would be an integer op)
csrrw x22, mscratch, x22
TY(x22); EQ(x22, x24)
csrrw x23, mscratch, zero
TY(x23); EQ(x23, a6)
STC(x29, x23, 0)
CLR; csrr x22, 0x5c0; CAUSE

# ---------------------------------------------------------------- S: what M stored comes back
li x22, 0
TO_S(s_fin, m_after_fin)
m_after_fin:
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
s_scr:
  # S1
  csrw sscratch, x26
  li t1, 0
  csrrw t1, sscratch, zero
  TY(t1)
  li t2, 64
  CLR; CDERIVE(x23, t1, t2); CAUSE
  TY(x23)
  csrr t2, sscratch
  EQI(t2, 0); TY(t2)
  # S2: rd = rs1
  csrw sscratch, x26
  CINCOFFSET(x22, x20, x0)       # a tagged copy of x20 (mv would be an integer op)
  csrrw x22, sscratch, x22
  TY(x22); EQ(x22, x24)
  csrrw t1, sscratch, zero
  TY(t1); EQ(t1, a6)
  CLR; ld x23, 8(t1); CAUSE
  # S3: csrr
  csrw sscratch, x26
  csrr t2, sscratch
  EQ(t2, x24); TY(t2)
  csrrw t1, sscratch, zero
  TY(t1)
  # S4: csrrs with a register that sets bits already set
  csrw sscratch, x26
  csrrs zero, sscratch, x24
  csrrw t1, sscratch, zero
  TY(t1); EQ(t1, x24)
  # S5: csrrci with 0
  csrw sscratch, x26
  csrrci t2, sscratch, 0
  TY(t2)
  csrrw t1, sscratch, zero
  TY(t1)
  # S6: csrrwi
  csrw sscratch, x26
  csrrwi zero, sscratch, 5
  csrrw t1, sscratch, zero
  TY(t1); EQI(t1, 5)
  # S7: csrw of an integer
  csrw sscratch, x26
  csrw sscratch, x24
  csrrw t1, sscratch, zero
  TY(t1)
  # S8: csrrs with a register that holds 0
  li t2, 0
  csrw sscratch, x26
  csrrs x22, sscratch, t2
  TY(x22)
  csrrw t1, sscratch, zero
  TY(t1); EQ(t1, x24)
  # S9: csrrc with a register that holds 0
  csrw sscratch, x26
  csrrc zero, sscratch, t2
  csrrw t1, sscratch, zero
  TY(t1); EQ(t1, x24)
  # S10: csrrs with x0, csrrsi with 0
  csrw sscratch, x26
  csrrs x22, sscratch, zero
  csrrsi x22, sscratch, 0
  csrrw t1, sscratch, zero
  TY(t1)
  # S11: csrrci with a bit already clear
  csrw sscratch, x26
  csrrci zero, sscratch, 1
  csrrw t1, sscratch, zero
  TY(t1); EQ(t1, x24)
  # S12: scapctl, then the srevroot control
  CLR; csrw 0x5c0, t2; CAUSE
  csrr t2, SREVROOT
  lla t1, tbl
  EQ(t2, t1)
  # park x20 for M
  csrw sscratch, x20
  ecall
.align 4
u_scr:
  csrrw x22, sscratch, x24
  csrw SREVROOT, zero
  ecall
.align 4
s_fin:
  CLR; LDC(x22, x29, 0); CAUSE
  TY(x22)
  CLR; ld x23, 8(x22); CAUSE
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
