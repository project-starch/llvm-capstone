# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-r60-alias.S -- ISSUES R-60 on Line B: an out-of-window cursor must not ALIAS (2026-11-26)
#-----------------------------------------------------------------------------
#
# The Line B port of P1's cap-alias-store.S (silchk-b1-2026-11-21). That test runs in M under CAPENTER, which the
# Sublet format deletes; on Line B the capability-checked data path is protected U (b5b2-uprot.S: in U with srevroot
# nonzero, every access needs a tagged base: an untagged one raises 24). Everything is minted through CSMINT (the
# silicon path). The table is b5b2-uprot.S's, and its root mint, verbatim.
# Geometry, as cap-alias-store.S's: the test capability x26 is a CDERIVE child over region, 0x1000 bytes, 0x4000-aligned
# (.align 14), so E = 0 and the window span is 0x4000. x22 is x26 moved by CINCOFFSET +0x4000 (in M): out of window, and
# the bounds decoded at that cursor (the true ones shifted by one span) contain it. The witness x23 is an independent
# CDERIVE child over [region + 0x4000, region + 0x5000), holding the sentinel 0x5a5a written by M.
# FIRST RUN VOID (2026-11-26, 3d9925be7; two construction errors of mine, corrected before this run): x26 was 0x4000 bytes
#   long, so E > 0 and region + 0x4000 lay INSIDE its window (no r60 FIRE; the store read 28, a plain bounds fault);
#   and every witness read took 25. Lengths are now P1's (0x1000).
# SECOND RUN (62d2f1318): r60 FIRE 1 (E=0, cursor 0x8000c000, start 0x80008000), the moved store 0x18, LCC 7; the
#   negative control: no FIRE, no trap at the store, LCC 1. But the witness still took 25: I had attributed it to slot
#   4, while the unit's SLU trace shows x26 = pool slot 5 and x23 = pool slot 6 (the two CSMINTs take 3 and 4), and tbl's
#   slot 6 was left retired by the page-initialisation loop. tbl slot 6 is now written live; slot 4 is uprot's again.
# THIRD RUN (c9f0cc854, 2026-11-26): AS PREDICTED, two-sided. Fixed: 0x5a5a 0x1234 0x0 0x1 0x18 0x5a5a 0x0 0x0 0x7 0x1,
#   one r60 FIRE (E=0, cursor 0x8000c000, start 0x80008000). Negative control (the predicate forced off, diff sha256
#   04b5ba4f4047a94c...): 0x5a5a 0x1234 0x0 0x0 0x0 0x9999 0x0 0x0 0x1 0x0, no FIRE -- the alias LANDS a span away.
#   Read with this tree's capprint-readings.py (sup-wt's copy is blind to U-mode writes).
# U (protected, srevroot = tbl) makes the reads and the store. Results are kept in a0-a6; ecall; M prints them.
#   A  0x5a5a  the sentinel through the witness (instrument control: else VOID)
#   B  0x1234  an in-window store then load through x26 (control: else VOID)
#   C  the trap count before the store through x22
#   D  the trap count after:   C+1 with the fix, C without it
#   E  the cause of that store: 0x18 (24, an untagged base) with the fix, 0 without it
#   F  0x5a5a  THE READING: the sentinel INTACT with the fix; 0x9999 without it (the alias landed)
#   a6 a7      0 0 (unused)
#   G  7       LCC type of x22, read in M: 7 = untagged with the fix; 1 (a tagged NONLIN) without it
#   then the trap count (ecalls not counted).
# SUP_TRACE: one "SUP r60 FIRE" line at the CINCOFFSET with the fix.
# PRE-REGISTERED (2026-11-26, written before any run), on b42c8c644's core (capstone-ariane 066c97947 + this test):
#   0x5a5a 0x1234 0x0 0x1 0x18 0x5a5a 0x0 0x0 0x7 0x1 (C = 0: nothing traps before the store), one r60 FIRE.
#   The negative control, the same tree with the R-60 predicate forced off (a local, uncommitted sim-only patch whose
#   diff is recorded with the reading): 0x5a5a 0x1234 0x0 0x0 0x0 0x9999 0x0 0x0 0x1 0x0, no FIRE. That is the arm's
#   proof that it can see the alias; without it a clean reading would say nothing.
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
MINT(x18, CAP_TYPE_NONLIN, _start, tbl_end, CAP_PERM_RWX)  # the root over the whole image
lla a3, region
lla a4, _start
sub a3, a3, a4
CINCOFFSET(t5, x18, a3)        # the root's cursor at region
lla t1, pg_rp0
li t0, 8
sd t0, 96(t1)                  # tbl slot 6: next 0, depth 1 (x23 is pool slot 6: the two CSMINTs take 3 and 4,
li t0, 10
sd t0, 104(t1)                 #   x26 then x23 take 5 and 6, by the unit's SLU trace); prev 5, generation 0: LIVE
li a1, 0x1000
CDERIVE(x26, t5, a1)           # x26/s10: the test capability over [region, region + 0x1000): E = 0
li a2, 0x4000
CINCOFFSET(t5, t5, a2)         # the root's cursor at region + 0x4000
CDERIVE(x23, t5, a1)           # x23/s7: the witness over [region + 0x4000, region + 0x5000)
li t0, 0x5a5a
lla t1, region
li t2, 0x4000
add t1, t1, t2
sd t0, 0(t1)                   # the sentinel, plain from M
CINCOFFSET(x22, x26, a2)       # x22/s6: x26 moved one span: out of window (R-60's predicate fires here)
li a7, 0; LCC(a7, x22, x1)     # G: the type (1 = a tagged child, 7 = untagged)
mv x31, a7

# ---------------------------------------------------------------- U, srevroot = tbl
lla t0, tbl
csrw SREVROOT, t0
li t0, 0x1800
csrc mstatus, t0               # MPP = U
lla t0, u_part
csrw mepc, t0
lla x30, m_after
mret
m_after:
csrw SREVROOT, x0
PRINT8
CAPPRINT(x31)
CAPPRINT(x28)
RVTEST_PASS

.align 4
u_part:
  CLR; li t1, 0; ld t1, 0(x23);    mv a0, t1        # A
  li t2, 0x1234; sd t2, 8(x26); li t1, 0; ld t1, 8(x26); mv a1, t1   # B
  mv a2, x28                                         # C
  CLR; li t2, 0x9999; sd t2, 0(x22); mv a4, x9       # E
  mv a3, x28                                         # D
  li t1, 0; ld t1, 0(x23);         mv a5, t1        # F
  li a6, 0
  li a7, 0
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
.align 14
region:
  .zero 0x8000
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
