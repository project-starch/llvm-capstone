# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-srevroot.S -- R-58's arm: does the caller get its OWN lifetime-table base back after a virtual context
#                  escapes, or does it inherit the CONTEXT's? (2026-10-19)
#-----------------------------------------------------------------------------
#
# DERIVED FROM va-runv.S by six asserted edits, so everything below the added readings is that arm's known-good
# scaffolding and its readings are the control: if any of them moves, this arm's modification broke something
# rather than measuring it. The edits are a second table page, the frame's srevroot word pointing at it, the
# caller's own write and two read-backs of 0x5C1 at S, the CSNODE variant, five added CAPPRINTs, and the zeroing
# of s3, s4 and s5.
#
# THE CLAIM UNDER TEST (R-58, filed source-derived and NOT yet measured). The switcher's register map was
# extended for item D with id 67 (srevroot), but the SAVE/RESTORE window was not: every walk and the exchange
# cover 0..66 (`capstone_dom_switcher.anvil:527-538`). Id 67 is written in exactly three places, all
# context-side: the ENTRY load from the frame's second word (`:170-173`), the ESCAPE save to the hidden word
# (`:481`) and the RESUME restore from it (`:315`). The only other writer is the CSR instruction
# (`csr_regfile.sv:2014`; the switcher's is `:2051`), and there is no caller-side shadow analogous to
# `runv_caller_priv_q`. So the caller's value should never come back.
#
# THE ARM. The caller at S writes 0x5C1 = A (its own table, `tbl`), the frame carries B (`tbl2`), the context
# escapes, and the caller reads 0x5C1 back. B CONFIRMS R-58; A REFUTES it.
#   -DSREV_ZERO     A = 0, so the reading also says whether the inherited value is the frame's word verbatim
#   -DSREV_CSNODE   a CSNODE at S before the entry and again after the escape. CSNODE is gated at decode on the
#                   table-namespace flag, so with A = 0 it should be illegal before and legal after -- the
#                   namespace flip, which is R-58's consequence rather than the defect itself
#
# PRE-REGISTERED READINGS, the five added ones, printed after va-runv's own:
#   s3  = tbl's physical base      A, as the caller wrote it (0 under -DSREV_ZERO). Also the CONTROL that S can
#                                  write 0x5C1 at all: an illegal-instruction trap here voids the arm.
#   s4  = tbl2's physical base     THE READING, if R-58 holds. tbl (or 0 under -DSREV_ZERO) refutes it.
#   tbl                            printed so the comparison needs no external value
#   tbl2                           likewise
#   s5  = 0                        the CSNODE variant is not compiled in by default
# Every va-runv reading above them must be unchanged.
#
# WHAT IT READ, run srv on d78265520 at S12_MEM_DELAY=12, with va-runv itself in the same load as the baseline.
# **R-58 IS CONFIRMED.** s3 read 0x80006000 (tbl, the caller's own A, so S can write 0x5C1 and the control
# holds) and **s4 read 0x80007000 -- tbl2, B, the CONTEXT's table.** Under -DSREV_ZERO, with A = 0, s4 read
# 0x80007000 again, so the caller inherits the frame's word verbatim rather than keeping anything of its own.
# -DSREV_CSNODE demonstrated the consequence: the SAME CSNODE instruction read mcause 2 (illegal at decode)
# BEFORE the entry and executed AFTER the escape, which is the namespace flip.
#
# AND THE ASSUMPTION THIS HEADER NAMED DID FAIL on the first run, exactly as written: tbl2 was left all zeros, so
# the context's load through x26 (node 2) raised cause 25 in the new namespace instead of reaching its ecall --
# +24 read 0x19 against va-runv's 0xb, the pc and address readings moved, and data[8] stayed 0. The srevroot
# readings never depended on the context completing (the escape is terminal either way and the switcher's walk is
# identical) but the CONTROL did, and a control differing from its baseline in two variables is not one.
#
# RE-RUN WITH tbl2 INITIALISED BYTE-IDENTICALLY TO tbl, run srv2 on 57c277f4e, and the control is now EXACT:
#   va-runv        0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb 0x0 0x9
#   va-srevroot    0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb  0x80006000 0x80007000
#                  0x80006000 0x80007000 0x0  0x0 0x9
# -- the first thirteen and the last two byte-identical to the baseline, so this arm is va-runv plus exactly one
# variable, and then:
#   s3 = 0x80006000   tbl, the caller's own A, read straight back: S can write 0x5C1 and the control holds
#   s4 = 0x80007000   tbl2 -- THE CONTEXT'S TABLE. **R-58 CONFIRMED.**
#   -DSREV_ZERO       s3 = 0x0 and s4 = 0x80007000 again: the caller inherits the frame's word VERBATIM
#   -DSREV_CSNODE     three leading 0x2 instead of one -- the legacy-C control, then the PRE-ENTRY CSNODE
#                     illegal at decode -- and s5 = 0x4, the POST-ESCAPE CSNODE succeeding with a node id. The
#                     namespace flip, illegal before and successful after, with every other reading still the
#                     baseline's.
#
# Below this line, va-runv.S's own description, unchanged:
#
# The monitor (M, capability mode) builds a RECYCLE lifetime table (record 2 the root), identity page tables with USER
# leaves (the S-mode code's page without U, since S cannot fetch from a user page), CPMP0 over the whole image (S-mode
# code is confined by the capability PMP in capability mode), and a 4 KiB frame (the caller's FP state at +2048): +0 satp, +8 the
# table, +56 options 0, +64 the PCC (a tagged LINEAR RWX capability over `ctx`), +480 the x26 slot (a tagged NONLIN RW
# over `data`, node 2). It writes 0x77 into data[0], then mret's into scalar S code, which runs CSRUNV rd=a2, rs1=the
# frame's physical address, rs2=0. The context (C mode, translated at U) loads data[0] through x26, stores it at
# data[8], sets a0 = 0x55 and x11 = the loaded word, and executes ecall: options bit 0 is clear, so the escape is
# terminal (kind 2, cause 11). S resumes at CSRUNV+4 with a2 = 2 and ecalls into M, which reads the frame through the
# caller's restored s9, then S runs action 1 (discard, a2 = 0). First, in M after CAPENTER (legacy C), CSRUNV is
# illegal: the control reads mcause 2 from the trap handler.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   mcause 2 (the legacy-C control); then after the first escape: a2 (the kind) 2; the caller's s2 0x1234 (restored
#   from the frame's second kilobyte); the frame's +16 kind 2, +24 cause 11, +32 pc minus ctx_ecall 0, +40 addr 0,
#   +48 raw a0 0x55; the PCC slot's cursor at +64 minus ctx_ecall 0; the x11 slot at +240 0x77; data[8] 0x77;
#   csupstatus 5 and csupcause 11 (the escape's CSR record, written beside the frame); then after action 1: a2 0;
#   mcause (the handler's own, the second ecall) 9.
# -DVA_QUANTUM: csupquantum 3000 and the context loops: the escape is kind 1 (the quantum): a2 1; s2 0x1234; +16 1;
#   +24 0 (QEMU's quantum event; csupcause keeps the mark); +32 minus ctx_loop 0; +40 0; +48 0 (a0 untouched); PCC cursor minus
#   ctx_loop 0; +240 0; data[8] 0 (never reached); csupstatus 3; csupcause 0x8000000000000010; then action 1: 0; 9.

# ITEM 3's ARMS (R-59), added 2026-10-20 as pure #ifdef additions, so the four arms above compile
# byte-identically and their readings stand without a re-run. The precheck accepts ANY non-zero word at
# dom_base+8 (`capstone_dom_switcher.anvil:383`) while the id-67 write masks the low 12 bits
# (`csr_regfile.sv:2051`), and a lifetime-table root is PAGE-ALIGNED BY DEFINITION -- the reference model's
# `cap_rev_table_open` has always refused `root == 0 || (root & 0xfff) != 0` (capstone-qemu b290f5656600,
# target/riscv/cap_rev_table.c:40, read at that ref). Ruled 2026-10-20: REFUSE, do not mask.
#   -DRA_SMALL    the frame's +8 is 0x800 -- non-zero, so it passes the precheck today, and masked to ZERO, so
#                 the context is ADMITTED and then runs outside the table namespace. This is R-59 itself.
#   -DRA_LOWBIT   the frame's +8 is tbl2 | 8 -- admitted today and masked back to tbl2, so it runs CORRECTLY
#                 by accident. The silent arm, and the reason masking is the wrong fix: a base one bit off
#                 would be rounded to a different page rather than refused.
# THE REFUSAL IS OBSERVABLE WITHOUT A NEW READING, which is why these ride this arm rather than a new one: the
# refuse path writes the frame's +16 = 2 and **+24 = 29** (`capstone_dom_switcher.anvil:429-431`), where an
# admitted context that ecalls leaves +16 = 2 and **+24 = 11**. Both set +16 = 2, so +24 is the discriminator
# and this arm already prints it. A refused CSRUNV also runs no SAVE walk, so the caller's srevroot is never
# touched and s4 must equal s3 whatever item 2 does.
#   PRE-REGISTERED, after item 3:
#     -DRA_SMALL   +24 = 0x1d (29), data[8] = 0 (the context never ran), s3 = s4 = 0x80006000
#     -DRA_LOWBIT  the same three values -- and this is the arm that FAILS LOUDEST if the fix tests the masked
#                  value instead of rejecting, because masking makes it indistinguishable from the control
#     the four arms above  UNCHANGED
#   BEFORE item 3 the same two arms read +24 = 0xb and data[8] = 0x77, i.e. admitted; -DRA_SMALL additionally
#   runs the context at table 0. Measuring that pre-state is what makes these a pair rather than an assertion.
#
# R-58 IS FIXED IN RTL, measured 2026-10-20 on the lane's item-2 commit (id 67 joins the caller's snapshot at
# +944, one branch of walk_advance). Every arm SUCCESS; three of the four readings exactly as pre-registered,
# the fourth REFUTED and recorded as such below.
#   va-srevroot     s4 0x80007000 -> 0x80006000, the caller's OWN tbl read straight back, and EXACTLY ONE of
#                   the twenty readings moved (128,993 cycles)
#   -DSREV_ZERO     s3 = 0x0 and s4 0x80007000 -> 0x0. The strongest of the set: a restored ZERO cannot be the
#                   frame's word, which is what the unfixed arm read verbatim (128,937)
#   -DSREV_ZERO -DSREV_CSNODE  FOUR leading 0x2 instead of three -- the POST-escape CSNODE is illegal at decode
#                   again, the caller being outside the table namespace with srevroot back at 0 -- and
#                   s5 0x4 -> 0x0 (129,027)
#   -DSREV_CSNODE alone (va-srevroot-n, new): s3 = s4 = 0x80006000, so the restore is confirmed here too, and
#                   the post-escape CSNODE stays LEGAL as predicted -- but **s5 reads 0x6 where 0x3 was
#                   pre-registered. THE PREDICTION IS REFUTED, and the arm's DESIGN was wrong besides.** With a
#                   non-zero own table BOTH CSNODEs are legal (the pre-entry one too, which the comment below
#                   still calls illegal), so s5 is the SECOND call's index from the SAME table and cannot name
#                   a table at all -- which was the whole point of the arm. The value is unexplained: two
#                   bump allocations from a next-unused of 3 give 4, not 6, so either CSNODE consumes more than
#                   one index per call (3,4 then 5,6, returning the second, which fits) or something else
#                   allocates from tbl in between. UNRESOLVED. The cheap settling reading is tbl's next-unused
#                   word at the end of the arm: 7 would confirm two indices per call. Nothing in item 2's
#                   acceptance depends on it -- the three arms above carry that, and this arm's s3 = s4 agrees.
#
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSRUNV(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x24, rd, rs1, rs2

#define CSNODE(rd, rs1)       .insn r 0x5B, 0x1, 0x54, rd, rs1, x0
#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define SREVROOT    0x5C1

#define PTE_LEAF_RWX  0xDF
#define PTE_LEAF_SRWX 0xCF
#define PTE_PTR       0x01

.option norvc

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0

li s3, 0
li s4, 0
li s5, 0
lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

# ---------------------------------------------------------------- the table: RECYCLE, next 3, record 2 the root
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
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

# ---------------------------------------------------------------- the CONTEXT's table, byte-identical to tbl
# The first run of this arm left tbl2 all zeros, and the context's load through x26 (node 2) then raised cause 25
# in the new namespace -- which the header had named as the assumption most likely to fail, and it did. The
# readings that matter did not depend on it, but the control did: va-runv's kind/cause/pc/data readings have to
# be reproduced EXACTLY for this arm to be that arm plus one variable. So tbl2 gets the same RECYCLE header and
# the same record 2 as the root, written through its own capability in s1.
MINT(s1, CAP_TYPE_NONLIN, tbl2, tbl2_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 64
sd t0, 0(s1)
li t0, 3
sd t0, 8(s1)
sd x0, 16(s1)
sd x0, 24(s1)
li t0, 32
CINCOFFSET(t1, s1, t0)
sd x0, 0(t1)
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)

# ---------------------------------------------------------------- data, and the context's capability over it (node 2)
MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)
li t0, 0x77
sd t0, 0(x26)
sd x0, 8(x26)

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
# the capability PMP: S-mode code is confined by CPMP in capability mode; grant the whole image
MINT(a1, CAP_TYPE_NONLIN, _start, pt_end, CAP_PERM_RWX)
CCSRRW(x0, CCSR_CPMP(0), a1)
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
mv s6, t0                      # the satp value, for the frame (s6 = x22; s10 is x26, the context's capability)

# ---------------------------------------------------------------- the frame
MINT(s9, CAP_TYPE_NONLIN, frame, frame_end, CAP_PERM_RW)
sd s6, 0(s9)                   # satp
#ifdef RA_SMALL
li t0, 0x800                   # R-59/item 3: non-zero but BELOW a page, so the id-67 write masks it to ZERO
#elif defined(RA_LOWBIT)
lla t0, tbl2
ori t0, t0, 8                  # R-59/item 3: a real base with a low bit set -- masked, it installs tbl2 silently
#else
lla t0, tbl2
#endif
sd t0, 8(s9)                   # srevroot := the CONTEXT's table, DELIBERATELY not the caller's (R-58's arm)
sd x0, 56(s9)                  # options 0
MINT(a2, CAP_TYPE_LIN, ctx, ctx_end, CAP_PERM_RWX)
STC(s9, a2, 64)                # the PCC slot
li t0, 480
CINCOFFSET(t1, s9, t0)
STC(t1, x26, 0)                # the x26 slot (+64 + 16 * 26)

#ifdef VA_QUANTUM
li t0, 3000
#else
li t0, 0x100000
#endif
csrw CSUPQUANTUM, t0

# ---------------------------------------------------------------- the control: CSRUNV in legacy C is illegal (2)
lla a0, frame
li a1, 0
CSRUNV(a2, a0, a1)             # traps: the handler prints mcause (2) and steps over it

# ---------------------------------------------------------------- into scalar S
li s2, 0x1234
li t0, 0x1800                  # MPP := 01 (S)
csrc mstatus, t0
li t0, 0x0800
csrs mstatus, t0
lla t0, s_code
csrw mepc, t0
mret

.align 12
s_code:
# R-58's arm. The caller sets its OWN lifetime-table base to A, enters a context whose frame carries B, and reads
# 0x5C1 back after the escape. B confirms the defect (nothing restores the caller's srevroot), A refutes it.
#ifdef SREV_ZERO
  csrw SREVROOT, x0            # A = 0: the namespace-flip variant
#else
  lla t0, tbl
  csrw SREVROOT, t0            # A = the caller's own table
#endif
  csrr s3, SREVROOT            # A as the caller left it; also the control that S can write 0x5C1 at all
#ifdef SREV_CSNODE
  CSNODE(s5, s8)               # BEFORE the entry: illegal at decode while the caller is out of the namespace
#endif
lla a0, frame
li a1, 0
CSRUNV(a2, a0, a1)             # the entry; S resumes here with a2 = the kind
  csrr s4, SREVROOT            # THE READING: B means the caller inherited the context's table
#ifdef SREV_CSNODE
  CSNODE(s5, s8)               # AFTER the escape: legal if the caller is now inside the table namespace
#endif
mv a3, a2
li a7, 1
ecall                          # to M: the readings
lla a0, frame
li a1, 1
CSRUNV(a2, a0, a1)             # action 1: discard
mv a3, a2
li a7, 2
ecall
1: j 1b

.align 12
trap_handler:
  csrr t0, mcause
  li t1, 2
  bne t0, t1, 3f
  CAPPRINT(t0)                 # 2: the legacy-C control
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
3:
  li t1, 1
  bne a7, t1, 4f
  CAPPRINT(a3)                 # the kind: 2 (quantum variant: 1)
  CAPPRINT(s2)                 # 0x1234: the caller's registers came back
  ld t2, 16(s9)
  CAPPRINT(t2)                 # +16 kind
  ld t2, 24(s9)
  CAPPRINT(t2)                 # +24 cause
  ld t2, 32(s9)
#ifdef VA_QUANTUM
  lla t3, ctx_loop
#else
  lla t3, ctx_ecall
#endif
  sub t2, t2, t3
  CAPPRINT(t2)                 # +32 pc: 0
  ld t2, 40(s9)
  CAPPRINT(t2)                 # +40 addr: 0
  ld t2, 48(s9)
  CAPPRINT(t2)                 # +48 raw a0
  ld t2, 64(s9)
  sub t2, t2, t3
  CAPPRINT(t2)                 # the PCC slot's cursor: 0
  ld t2, 240(s9)
  CAPPRINT(t2)                 # the x11 slot
  ld t2, 8(x26)
  CAPPRINT(t2)                 # data[8]
  csrr t2, CSUPSTATUS
  CAPPRINT(t2)                 # 5 / 3
  csrr t2, CSUPCAUSE
  CAPPRINT(t2)                 # 11 / the quantum mark
  CAPPRINT(s3)                 # A, as the caller wrote it: tbl's physical base (0 under SREV_ZERO)
  CAPPRINT(s4)                 # THE READING: tbl2 confirms R-58, tbl (or 0) refutes it
  lla t2, tbl
  CAPPRINT(t2)                 # A's address, so the comparison needs no external value
  lla t2, tbl2
  CAPPRINT(t2)                 # B's address
  CAPPRINT(s5)                 # the CSNODE variant's node id, 0 when it is not compiled in
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
4:
  CAPPRINT(a3)                 # 0: the discard
  csrr t0, mcause
  CAPPRINT(t0)                 # 9
  CAP_PASS(s11)

.align 6
ctx:
#ifdef VA_QUANTUM
ctx_loop:
  j ctx_loop
#else
  ld t1, 0(x26)
  sd t1, 8(x26)
  li a0, 0x55
  mv x11, t1
ctx_ecall:
  ecall
  j ctx_ecall
#endif
ctx_end:

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
data:
  .zero 4096
data_end:
.align 12
tbl:
  .zero 4096
tbl_end:
.align 12
tbl2:
  .zero 4096
tbl2_end:
.align 12
frame:
  .zero 4096
frame_end:
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
