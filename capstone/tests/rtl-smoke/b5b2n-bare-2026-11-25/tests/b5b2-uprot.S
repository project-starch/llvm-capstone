# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-uprot.S -- B5b-2 step 2: the LSU admission gate by privilege (protected U, S rule 1, M plain) (2026-11-24)
#-----------------------------------------------------------------------------
#
# The reference (capstone-qemu 9289114e06): U is protected exactly when srevroot is nonzero, and every data access of
# protected U goes through a reference (an untagged base raises 24, a stale one 25, out of bounds 28, in that order:
# op_helper.c capstone_check_access checks the tag, then liveness, then type, rights and bounds). In S with srevroot
# nonzero, a TAGGED base is checked the same way (rule 1) and an untagged one is plain. M is plain.
#
# Table: sublet-ns.S's paged Sublet table exactly, with its SL_F4 record (slot 4 at generation 2^30): slot 2 the root,
# 5, 4 (stale for generation 0), 3 (live). Capabilities by cderive in the physical pool, whose slots are the same
# numbers: x26/s10 = {0,3} over data (live in tbl), x27/s11 = {0,4} over data (stale in tbl). data[0] = 0x77.
# Translation is bare (satp 0) in every mode; PMP entry 0 is NAPOT over everything, RWX, so U and S reach memory.
# Every U and S trap goes to M (medeleg 0). The handler records mcause in x9/s1 and mtval in x19/s3, counts the trap in
# x28/t3 and resumes at mepc + 4; an ecall (mcause 8 or 9) instead returns to M's continuation at x30/t5.
# x24/s8 holds data's address as an INTEGER; x20/s4 and x21/s5 are x26 and x27 moved to data + 4088, so ld 8(x20) and
# ld 8(x21) are one past the bounds (data + 4096). Results go to x10-x17 (a0-a7) and M prints them after each phase.
#
# Phase 1, U with srevroot = tbl (protected U):
#   a0 cause of  ld x26+0  (live, in bounds)       a1 its value
#   a2 cause of  ld x27+0  (stale)                 a3 cause of  ld x24+0  (an untagged base)
#   a4 cause of  ld 8(x20) (out of bounds)         a5 cause of  ld 8(x21) (stale AND out of bounds)
#   a6 cause of  sd 0x99 -> x26+8                  a7 value of  ld x26+8
# Phase 2, U with srevroot = 0 (U unprotected):
#   a0, a1 cause and value of  ld x24+0           a2, a3 cause and value of  ld x27+0
# Phase 3, S with srevroot = tbl (rule 1):
#   a0 cause of  ld x27+0 (tagged, stale)          a1, a2 cause and value of  ld x24+0 (untagged: plain)
#   a3 cause of  ld 8(x20) (tagged, out of bounds)
# Phase 4, M with srevroot = tbl:  a0, a1 cause and value of  ld x27+0
# Then the trap count (ecalls not counted).
#
# PRE-REGISTERED (2026-11-24, written before any run), Sublet build:
#   reference / step 2 (b5b2-step2):
#     phase 1: 0 0x77 | 0x19 0x18 | 0x1c 0x19 | 0 0x99      (24 = 0x18, 25 = 0x19, 28 = 0x1c)
#     phase 2: 0 0x77 | 0 0x77
#     phase 3: 0x19 | 0 0x77 | 0x1c
#     phase 4: 0 0x77
#     trap count 6
#   base (b5b2-step1 c36facb5a, capmode 0 so nothing is checked):
#     phase 1: 0 0x77 | 0 0 | 0 0 | 0 0x99   (every a-register but a1 and a7 is a cause; the data + 4096 loads read the
#              zero page after data)
#     phase 2: 0 0x77 | 0 0x77   phase 3: 0 | 0 0x77 | 0   phase 4: 0 0x77   trap count 0
#   a5 is the order arm: the reference checks liveness before bounds, so stale and out of bounds is 25, not 28.
# -DUP_MIS (the claim-auditor's F1, added 2026-11-24 before any run of it): four more protected-U arms, misaligned
#   lw, causes in x22/s6, x23/s7, x25/s9 and x31/t6, printed after phase 1's eight:
#     lw 1(x24) untagged      lw 9(x20) out of bounds      lw 1(x26) live, in bounds      lw 1(x27) stale
#   PRE-REGISTERED: the reference 0x18 0x1c 0x4 0x19 (the capability first; a passing check leaves the misalignment).
#   b5b2-step2 BEFORE the F1 fix and the base: 0x4 0x4 0x4 0x4 (the misalignment outranks everything). After the fix:
#   the reference's. Trap count 6 + 4 = 10 on the step-2 trees, 4 on the base. No firmware runs in simulation: the
#   handler records the 4 and skips the access, so this arm sees the cause and not the emulation.
# B5b-2 (2026-11-24, the semantics owner's ruling): uses the capcreate family, to be ported to CSMINT.

# LEGACY SCAFFOLD (B5b-2 step 9, 2026-11-25): this test mints with the custom-3 debug family, which decodes under the
# Sublet format only in simulation builds with CAPSTONE_SIM_DEBUG_MINT. -DCAP_MINT_CSMINT (the bare kits, silicon)
# mints through CSMINT instead, which allocates its own lifetime record, so pool ids shift by one there. New tests
# mint with CSMINT only.
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT    0x5C1

#ifdef CAP_MINT_CSMINT
/* CSMINT through a descriptor {base, end, perms 0x10f, cursor = base, 0, 0} and a granule private to this use (in
   .data), then an M ldc of the granule (M through a plain base keeps the tag). Clobbers a0, a3, a4, a5. */
#define CS_MINT_INTO(REG, LO, HI) \
  .pushsection .data; .balign 16; 77: .zero 64; .popsection; \
  lla a0, 77b; lla a3, LO; sd a3, 0(a0); lla a4, HI; sd a4, 8(a0); li a5, 0x10f; sd a5, 16(a0); \
  sd a3, 24(a0); sd x0, 32(a0); sd x0, 40(a0); addi a5, a0, 48; .insn r 0x5B, 0x1, 0x50, a3, a5, a0; LDC(REG, a5, 0)
/* the Sublet format stores no type and no rights: a loaded capability decodes NONLIN with every right */
#define MINT(REG, TYPE, LO, HI, PERM) CS_MINT_INTO(REG, LO, HI)
#else
#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)
#endif

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

# ---------------------------------------------------------------- phase 1: U, srevroot = tbl
lla t0, tbl
csrw SREVROOT, t0
li t0, 0x1800
csrc mstatus, t0               # MPP = U
lla t0, u_phase1
csrw mepc, t0
lla x30, m_after1
mret
m_after1:
PRINT8
#ifdef UP_MIS
CAPPRINT(x22); CAPPRINT(x23); CAPPRINT(x25); CAPPRINT(x31)
#endif

# ---------------------------------------------------------------- phase 2: U, srevroot = 0
csrw SREVROOT, x0
li t0, 0x1800
csrc mstatus, t0
lla t0, u_phase2
csrw mepc, t0
lla x30, m_after2
mret
m_after2:
CAPPRINT(a0); CAPPRINT(a1); CAPPRINT(a2); CAPPRINT(a3)

# ---------------------------------------------------------------- phase 3: S, srevroot = tbl
lla t0, tbl
csrw SREVROOT, t0
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0               # MPP = S
lla t0, s_phase3
csrw mepc, t0
lla x30, m_after3
mret
m_after3:
CAPPRINT(a0); CAPPRINT(a1); CAPPRINT(a2); CAPPRINT(a3)

# ---------------------------------------------------------------- phase 4: M, srevroot = tbl (still)
CLR
ld t1, 0(x27)
mv a0, x9
mv a1, t1
CAPPRINT(a0); CAPPRINT(a1)
csrw SREVROOT, x0
CAPPRINT(x28)
RVTEST_PASS

# ---------------------------------------------------------------- the U and S code
.align 4
u_phase1:
  CLR; li t1, 0; ld t1, 0(x26);    mv a0, x9; mv a1, t1
  CLR;           ld t1, 0(x27);    mv a2, x9
  CLR;           ld t1, 0(x24);    mv a3, x9
  CLR;           ld t1, 8(x20);    mv a4, x9
  CLR;           ld t1, 8(x21);    mv a5, x9
  CLR; li t2, 0x99; sd t2, 8(x26); mv a6, x9
  CLR; li t1, 0; ld t1, 8(x26);    mv a7, t1
#ifdef UP_MIS
  CLR;           lw t1, 1(x24);    mv x22, x9
  CLR;           lw t1, 9(x20);    mv x23, x9
  CLR;           lw t1, 1(x26);    mv x25, x9
  CLR;           lw t1, 1(x27);    mv x31, x9
#endif
  ecall
.align 4
u_phase2:
  CLR; li t1, 0; ld t1, 0(x24);    mv a0, x9; mv a1, t1
  CLR; li t1, 0; ld t1, 0(x27);    mv a2, x9; mv a3, t1
  ecall
.align 4
s_phase3:
  CLR;           ld t1, 0(x27);    mv a0, x9
  CLR; li t1, 0; ld t1, 0(x24);    mv a1, x9; mv a2, t1
  CLR;           ld t1, 8(x20);    mv a3, x9
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
  TEST_DATA
RVTEST_DATA_END
