# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-sref.S -- B5b-2 step 2b: supervisor references (LDC/STC of S and M through a plain base keep the tag; LCC 8 in S)
#-----------------------------------------------------------------------------
#
# The semantics owner's ruling (2026-11-24) on capstone-qemu 9289114e06 with x-capstone-supervisor-refs
# (trans_capstone.c.inc plain_base_refs, gen_csldc_body; op_helper.c LCC case 8): an LDC/STC is CHECKED only in a
# capability context (protected U; S with srevroot != 0 and a tagged base). Any other LDC/STC of S or M is a plain access
# of the mode that keeps the tag. Unprotected U has no LDC/STC (illegal, 2). LCC selector 8 in S answers the operand's
# lifetime id, generation << 31 | slot.
# Table and capabilities: b5b2-uprot.S's, bare translation everywhere. x26/s10 = {0,3} live over data, x27/s11 = {0,4},
# x20/s4 = x26 moved to data + 4088 (LDC 8(x20) is one past its bounds), x24/s8 data as an integer, x29/t4 refbox as an
# integer (outside data), x18/s2 the root {0,2}. LDC destinations are cleared before each phase. Traps go to M; the
# handler records mcause in x9/s1, counts in x28/t3 (zeroed after the setup), resumes at mepc + 4; ecall returns to M at
# x30/t5. LCC(rd, rs, x1) is selector 1, the type: 1 = a tagged NONLIN, 7 = untagged; x2 the cursor; x8 the id.
# Prints:
#   M (srevroot = tbl): a0 cause of stc x26 -> 0(x29)   a1 cause of ldc x21 <- 0(x29)   a2 LCC type of x21
#                       a3 (x21's cursor == data)        a4 cause of ldc x22 <- 8(x20) (a tagged base, out of bounds)
#   S (srevroot = tbl): a0 cause of stc x27 -> 16(x29)  a1 cause of ldc x23 <- 16(x29)  a2 LCC type of x23
#                       a3 LCC 8 of x23   a4 LCC 8 of x26   a5 cause of ldc x22 <- 8(x20) (rule 1: checked)
#                       a6 LCC 8 of x25, x25 = cderive of x18 over 64 bytes in tbl
#                       then (M) a6's slot, the generation tbl's record holds for it, and (a6 == gen << 31 | slot)
#   S (srevroot = 0):   a0 cause of ldc x22 <- 8(x20) (a tagged base, srevroot 0: plain)
#   U (srevroot = 0):   a0 cause of ldc x22 <- 0(x26)
#   U (srevroot = tbl): a0 cause of ldc x22 <- 0(x24) (integer)   a1 cause of ldc x22 <- 0(x26)
#   the trap count
# PRE-REGISTERED (2026-11-24, written before any run), Sublet build:
#   step 2b:  M 0 0 1 1 0 | S 0 0 1 4 3 0x1c <id> | <slot> <gen> 1 | S0 0 | U0 2 | U1 0x18 0 | traps 3
#             (<id> = <gen> << 31 | <slot>: the slot and generation are recorded, not predicted; the flag is the check)
#   base (b5b2-step2 99d7ef22c): M 0x18 0x18 7 0 0x1c | S 0x18 0x18 7 0 0 0x1c 0 | 0 <rec gen of slot 0> 0 | S0 0x1c |
#             U0 0 | U1 0x18 0 | traps 10 (M: stc, ldc, the lcc cursor of an untagged x21, ldc; S: stc, ldc, lcc 8 of
#             an untagged x23, ldc; S0: ldc; U1: ldc)
# FIRST RUN (e68d49a9c): the base read its whole vector, but the trap count read 0x100000001 on both trees: the M
#   cross-check used t3 = x28 (the counter), the class of memory each-phase-sets-its-own-preconditions. The cross-check
#   now uses x31 and x16. Step 2b's first cut read LCC 8 = 0 (its override missed: LCC answers in its dispatch cycle),
#   and the CSRUNV-context tests lost their LDC checks (contexts run at M). Both are fixed in the RTL, not here.
# -DSR_GEN (the step-2b claim-auditor's point: LCC 8's generation bits were exercised only at 0; added before any run of
#   it): tbl's slot 6 is the only free record, at generation 0xC0000001, so the cderive in S takes it. PRE-REGISTERED on
#   step 2b: every other value as above; the cross-check reads slot 6, the record's generation G (0xc0000001 if the
#   allocation keeps the free record's generation; recorded, not predicted), flag 1; and LCC 8 = G << 31 | 6, so bits
#   62..61 of the id are set (0x60000000_80000006 for G = 0xc0000001). Run on step 2b only: the base answers LCC 8 with 0.
# B5b-2 (2026-11-24, the semantics owner's ruling): uses the capcreate family, to be ported to CSMINT.

# LEGACY SCAFFOLD (B5b-2 step 9, 2026-11-25): this test mints with the custom-3 debug family, which decodes under the
# Sublet format only in simulation builds with CAPSTONE_SIM_DEBUG_MINT. -DCAP_MINT_CSMINT (the bare kits, silicon)
# mints through CSMINT instead, which allocates its own lifetime record, so pool ids shift by one there. New tests
# mint with CSMINT only.
# KIT BUILD (-DCAP_MINT_CSMINT, 2026-11-25): CSMINT's root is pool slot 3; s8 is a plain address (no second mint), the
#   table is re-keyed one up (3 -> 6 -> 5 -> 4, next 7; SR_GEN's free record slot 7, next 8), and a key check ends the
#   run with the sentinel 0x6b657962616420 if the root is not {0,3}. PRE-REGISTERED, the kit images (positions):
#   sref:     as the in-tree vector except 9: 0x5 (LCC 8 of x23 = {0,5}), 10: 0x4 ({0,4}), 12: 0x7 (a6, slot 7),
#             13: 0x7 (its slot), 14: 0x0 (generation), 15: 0x1 (the cross-check)
#   sref-gen: the same, except 12: 0x6000000080000007 (0xC0000001 << 31 | 7), 14: 0xc0000001, 15: 0x1
#   with -DKIT_ROOT_SLOT=4 (the deliberate skew): the sentinel alone.
#   The first kit images (b5b2n, before the re-key) read a6 = 0 / 0xffffffff, flag 0: tables keyed to CAPCREATE-era slots.
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
#ifdef CAP_MINT_CSMINT
lla s8, tbl                    # the kit build: M is integer, the table is written plainly (no second mint)
#else
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
#endif
li t0, 3
slli t0, t0, 62
ori t0, t0, 512
sd t0, 0(s8)                   # capacity 512 | RECYCLE | PAGED
#ifdef CAP_MINT_CSMINT
li t0, 7
sd t0, 8(s8)                   # next 7 (the kit's re-key)
#else
li t0, 6
sd t0, 8(s8)                   # next 6
#endif
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
#ifdef CAP_MINT_CSMINT
lla t1, pg_rp0
# the kit build (CAP_MINT_CSMINT): CSMINT's root takes pool slot 3, so the list is re-keyed one up, 3 -> 6 -> 5 -> 4
li t0, 6
slli t0, t0, 34
sd t0, 48(t1)                  # slot 3: next 6, depth 0 (the root)
sd x0, 56(t1)                  #         generation 0, prev 0
li t0, 5
slli t0, t0, 34
ori t0, t0, 8
sd t0, 96(t1)                  # slot 6: next 5, depth 1
li t0, 6
sd t0, 104(t1)                 #         prev 3
li t0, 4
slli t0, t0, 34
ori t0, t0, 8
sd t0, 80(t1)                  # slot 5: next 4, depth 1
li t0, 12
li t2, 1
slli t2, t2, 62
or t0, t0, t2
sd t0, 88(t1)                  #         prev 6, generation 2^30 (stale for the generation-0 id x27 carries)
li t0, 8
sd t0, 64(t1)                  # slot 4: next 0, depth 1
li t0, 10
sd t0, 72(t1)                  #         prev 5
#else
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
#endif

# ---------------------------------------------------------------- the capabilities, by cderive in the physical pool
MINT(x18, CAP_TYPE_NONLIN, _start, tbl_end, CAP_PERM_RWX)  # pool node 2: {0,2}
#ifdef CAP_MINT_CSMINT
#ifndef KIT_ROOT_SLOT
#define KIT_ROOT_SLOT 3
#endif
# the kit's key check (the semantics owner's condition 1): the root CSMINT returned must be the slot and generation the
# re-keyed tables assume, {0, KIT_ROOT_SLOT}. Word 1 of the minted granule (a5) is generation << 32 | slot << 1 | bounds[26].
# On a mismatch the run ends with the sentinel alone, never a vector. -DKIT_ROOT_SLOT=4 is the deliberate skew.
ld a3, 8(a5)
srli a4, a3, 1
li a0, 0x7fffffff
and a4, a4, a0
li a0, KIT_ROOT_SLOT
bne a4, a0, kit_key_bad
srli a4, a3, 32
bnez a4, kit_key_bad
j kit_key_ok
kit_key_bad:
li a0, 0x6b657962616420        # the sentinel
CAPPRINT(a0)
RVTEST_PASS
kit_key_ok:
#endif
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

#ifdef SR_GEN
# a free record with a high generation for the cderive to take: slot 6, generation 0xC0000001 (bits 31, 30 and 0), the
# only free entry (header quadword 1 = {count 1, head 6}); next 7. Plain M stores (s8 was the table's capability, and
# x24 = s8 now holds data's address).
#ifdef CAP_MINT_CSMINT
lla t0, tbl
li t1, 8
sd t1, 8(t0)                   # next 8 (the kit's re-key: the free record is slot 7)
li t1, 1
slli t1, t1, 32
ori t1, t1, 7
sd t1, 16(t0)                  # free list: count 1, head 7
lla t0, pg_rp0
sd x0, 112(t0)                 # slot 7 word 0: next 0, depth 0
li t1, 0xC0000001
slli t1, t1, 32
sd t1, 120(t0)                 # slot 7 word 1: generation 0xC0000001, prev 0
#else
lla t0, tbl
li t1, 7
sd t1, 8(t0)                   # next 7
li t1, 1
slli t1, t1, 32
ori t1, t1, 6
sd t1, 16(t0)                  # free list: count 1, head 6
lla t0, pg_rp0
sd x0, 96(t0)                  # slot 6 word 0: next 0, depth 0
li t1, 0xC0000001
slli t1, t1, 32
sd t1, 104(t0)                 # slot 6 word 1: generation 0xC0000001, prev 0
#endif
#endif
lla x29, refbox
li x21, 0
li x22, 0
li x23, 0
li x25, 0
li x28, 0                      # the trap count, after every setup block

# ---------------------------------------------------------------- M, srevroot = tbl
lla t0, tbl
csrw SREVROOT, t0
CLR; STC(x29, x26, 0);  mv a0, x9
CLR; LDC(x21, x29, 0);  mv a1, x9
li a2, 0; LCC(a2, x21, x1)
li a3, 0; LCC(a3, x21, x2); sub a3, a3, x24; seqz a3, a3
CLR; LDC(x22, x20, 8);  mv a4, x9
CAPPRINT(a0); CAPPRINT(a1); CAPPRINT(a2); CAPPRINT(a3); CAPPRINT(a4)

# ---------------------------------------------------------------- S, srevroot = tbl
lla t0, tbl
csrw SREVROOT, t0
li x22, 0
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0               # MPP = S
lla t0, s_tbl
csrw mepc, t0
lla x30, m_after_s
mret
m_after_s:
CAPPRINT(a0); CAPPRINT(a1); CAPPRINT(a2); CAPPRINT(a3); CAPPRINT(a4); CAPPRINT(a5); CAPPRINT(a6)
li t0, 0x7fffffff
and t1, a6, t0                 # slot
srli t2, a6, 31                # generation
lla t0, pg_rp0
slli x31, t1, 4                # x31/t6 and x16/a6 only: t3 is x28 (the trap count), t4 is x29 (refbox)
add t0, t0, x31
ld x31, 8(t0)
srli x31, x31, 32              # the record's generation
sub x16, x31, t2
seqz x16, x16
CAPPRINT(t1); CAPPRINT(x31); CAPPRINT(x16)

# ---------------------------------------------------------------- S, srevroot = 0
csrw SREVROOT, x0
li x22, 0
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0
lla t0, s_zero
csrw mepc, t0
lla x30, m_after_s0
mret
m_after_s0:
CAPPRINT(a0)

# ---------------------------------------------------------------- U, srevroot = 0
csrw SREVROOT, x0
li x22, 0
li t0, 0x1800
csrc mstatus, t0               # MPP = U
lla t0, u_zero
csrw mepc, t0
lla x30, m_after_u0
mret
m_after_u0:
CAPPRINT(a0)

# ---------------------------------------------------------------- U, srevroot = tbl
lla t0, tbl
csrw SREVROOT, t0
li x22, 0
li t0, 0x1800
csrc mstatus, t0
lla t0, u_tbl
csrw mepc, t0
lla x30, m_after_u1
mret
m_after_u1:
CAPPRINT(a0); CAPPRINT(a1)
csrw SREVROOT, x0
CAPPRINT(x28)
RVTEST_PASS

.align 4
s_tbl:
  CLR; STC(x29, x27, 16); mv a0, x9
  CLR; LDC(x23, x29, 16); mv a1, x9
  li a2, 0; LCC(a2, x23, x1)
  li a3, 0; LCC(a3, x23, x8)
  li a4, 0; LCC(a4, x26, x8)
  CLR; LDC(x22, x20, 8);  mv a5, x9
  li t1, 64; CDERIVE(x25, x18, t1)
  li a6, 0; LCC(a6, x25, x8)
  ecall
.align 4
s_zero:
  CLR; LDC(x22, x20, 8);  mv a0, x9
  ecall
.align 4
u_zero:
  CLR; LDC(x22, x26, 0);  mv a0, x9
  ecall
.align 4
u_tbl:
  CLR; LDC(x22, x24, 0);  mv a0, x9
  CLR; LDC(x22, x26, 0);  mv a1, x9
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
  TEST_DATA
RVTEST_DATA_END
