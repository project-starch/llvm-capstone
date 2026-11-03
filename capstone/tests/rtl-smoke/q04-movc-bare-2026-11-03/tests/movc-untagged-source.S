# See LICENSE for license details.

#*****************************************************************************
# movc-untagged-source.S -- batch item 5, Q-04 (b): MOVC must KEEP an untagged source (2026-10-20)
#-----------------------------------------------------------------------------
#
# THE RULE (the lead's decision 2026-10-14, option 1; plans/2026-10-14-movc-untagged-source.md). With rd != rs1,
# MOVC writes x[rs1] to x[rd] and writes cnull to x[rs1] ONLY if x[rs1] is a TAGGED capability whose type is not
# NONLIN. Untagged data (NOT_CAP, zero or not) and tagged NONLIN stay. rd == rs1 is a no-op. No exception.
#
# THE CHANGE UNDER TEST. capstone_flu_unit.anvil MOVC: NOT_CAP joins NONLIN in the preserve branch. Before it, the
# The else branch sends create_cnull() as the new rs1 for every non-NONLIN source, NOT_CAP included -- the field's C-32
# stops (cpython's `movc s2, a3; jalr a3` with a3 = 0).
#
# STRUCTURE. Every case runs and every reading is taken (CAPPRINT into the .iss log, integer values into saved
# registers) BEFORE any check, so a run on RTL WITHOUT the change still prints the whole matrix and then fails at
# the first check -- the negative arm reads the same file. The checks are on INTEGER values only, because a
# capability's TYPE cannot be branched on here (an LCC type query raises on NOT_CAP, see linear-clear-audit.S), so
# the tagged rows are observational through CAPPRINT, exactly as linear-clear-audit's arms 0-2 are.
#
# This file goes through cpp: instruction names in comments are written bare, never as a macro call.
#
# THE CASES, in order (the plan's step-2 list):
#   A  NOT_CAP non-zero, rd != rs1, and the VERY NEXT instructions read rs1 and then rd (forwarding and the FLU
#      result pack's dual write-back)                                       s2 = s3 = 0x1122334455667788 expected
#   B  NOT_CAP zero, rd != rs1                                              s4 = s5 = 0
#   C  NOT_CAP, rd == rs1                                                   s6 = 0x55
#   D  each consumed tagged type, rd != rs1: LIN, REV, UNINIT, SEALED, SEALEDRET, EXIT
#                                     the source prints type NOT_CAP (0), cursor 0; the destination prints that type
#   E  NONLIN, rd != rs1 (the control that a source CAN survive)            the source prints type NONLIN (2)
#   F  rd = x0 with a NOT_CAP source (kept: s7 = 0x77), a LIN source (prints NOT_CAP, destroyed) and a NONLIN
#      source (prints NONLIN, kept); x0 reads 0 afterwards (t2 = 0)
#   G  rs1 = x0: rd gets cnull                                              s8 = 0
#
# PRE-REGISTERED READINGS (written before any run).
#   WITH the change (the batch tree):
#     CAPPRINT order: D: (LIN src, LIN dst), (REV src, REV dst), (UNINIT src, UNINIT dst), (SEALED src, SEALED dst),
#       (SEALEDRET src, SEALEDRET dst), (EXIT src, EXIT dst) -- each src type 0 with cursor 0, each dst the named
#       type; E: NONLIN src type 2; F: LIN src type 0, NONLIN src type 2; x0 type 0.
#     Every check passes and tohost reads 1. The harness's SUCCESS alone is NOT the reading (it also prints
#     SUCCESS at the cycle limit): the reading is tohost = 1 with a retired count well below the ceiling, and the
#     CAPPRINT lines above.
#   WITHOUT the change (the NEGATIVE ARM, the same file on c7e6f604f, the field RTL):
#     every CAPPRINT line identical to the above (rows D, E, the LIN and NONLIN rows of F and row G do not change),
#     and the run FAILS with TESTNUM 1: s2, the NOT_CAP non-zero source of case A, reads 0 (nulled). Cases C and G
#     would pass and case F's NOT_CAP row would fail (TESTNUM 6), but the first failing check stops the run.
#   The pair (fixed passes, field fails at TESTNUM 1, every tagged row identical on both) is the reading. A field
#   run that PASSES means the instrument cannot see the bug, and the fixed reading proves nothing.
#
# CONTROLS THAT MUST FIRE. Case D's first row (LIN consumed) is the instrument control: it proves CAPPRINT shows a
# cleared source, as linear-clear-audit's arm 0 does. Case E is the opposite control: a NONLIN source survives.
# The negative arm is the control for the integer checks.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:

# Helpers a0, a2, a3, a4 never hold a capability (an ordinary instruction does not clear a register's capability
# metadata, and a stale tag turns a plain-integer operand of a capability instruction into UNEXPECTED_OPERAND).
#define MKCAP(CREG, CTYPE)   \
  CAPCREATE(CREG);           \
  li  a0, CTYPE;             \
  lla a2, test_buf_start;    \
  lla a3, test_buf_end;      \
  li  a4, CAP_PERM_RWX;      \
  CAPTYPE(CREG, a0);         \
  CAPBOUND(CREG, a2, a3);    \
  CAPPERM(CREG, a4);

# ---------------------------------------------------------------- A: NOT_CAP non-zero, rd != rs1, forwarding
li   s2, 0x1122334455667788
MOVC(s3, s2)
add  t0, s2, x0           # rs1 read in the very next instruction
add  t1, s3, x0           # then rd

# ---------------------------------------------------------------- B: NOT_CAP zero, rd != rs1
li   s4, 0
MOVC(s5, s4)

# ---------------------------------------------------------------- C: NOT_CAP, rd == rs1
li   s6, 0x55
MOVC(s6, s6)

# ---------------------------------------------------------------- D: every consumed tagged type
MKCAP(a5, CAP_TYPE_LIN)
MOVC(a6, a5)
CAPPRINT(a5)              # D LIN: src type 0 (consumed) -- the INSTRUMENT CONTROL
CAPPRINT(a6)              # D LIN: dst type LIN
MKCAP(a5, CAP_TYPE_REV)
MOVC(a6, a5)
CAPPRINT(a5)              # D REV: src type 0
CAPPRINT(a6)              # D REV: dst type REV
MKCAP(a5, CAP_TYPE_UNINIT)
MOVC(a6, a5)
CAPPRINT(a5)              # D UNINIT: src type 0
CAPPRINT(a6)              # D UNINIT: dst type UNINIT
MKCAP(a5, CAP_TYPE_SEALED)
MOVC(a6, a5)
CAPPRINT(a5)              # D SEALED: src type 0
CAPPRINT(a6)              # D SEALED: dst type SEALED
MKCAP(a5, CAP_TYPE_SEALEDRET)
MOVC(a6, a5)
CAPPRINT(a5)              # D SEALEDRET: src type 0
CAPPRINT(a6)              # D SEALEDRET: dst type SEALEDRET
MKCAP(a5, CAP_TYPE_EXIT)
MOVC(a6, a5)
CAPPRINT(a5)              # D EXIT: src type 0
CAPPRINT(a6)              # D EXIT: dst type EXIT

# ---------------------------------------------------------------- E: NONLIN kept (the opposite control)
MKCAP(a7, CAP_TYPE_NONLIN)
MOVC(s9, a7)
CAPPRINT(a7)              # E NONLIN: src type 2, unchanged

# ---------------------------------------------------------------- F: rd = x0
li   s7, 0x77
MOVC(x0, s7)              # NOT_CAP source: kept
MKCAP(s10, CAP_TYPE_LIN)
MOVC(x0, s10)
CAPPRINT(s10)             # F LIN: src type 0 (destroyed)
MKCAP(s11, CAP_TYPE_NONLIN)
MOVC(x0, s11)
CAPPRINT(s11)             # F NONLIN: src type 2 (kept)
add  t2, x0, x0           # x0 still reads 0
CAPPRINT(x0)              # F: x0 type 0

# ---------------------------------------------------------------- G: rs1 = x0
li   s8, 0x99
MOVC(s8, x0)

#ifdef BOARD_INTS
# Board kit (2026-11-03, r58-style bare arm): the integer readings are RECORDED before any check, so a run on either
# semantics returns its whole vector instead of stopping at the first failing check: s2, t0, t1 (A), s4, s5 (B),
# s6 (C), s7, t2 (F), s8 (G). The default build does not define BOARD_INTS and is unchanged.
CAPPRINT(s2)
CAPPRINT(t0)
CAPPRINT(t1)
CAPPRINT(s4)
CAPPRINT(s5)
CAPPRINT(s6)
CAPPRINT(s7)
CAPPRINT(t2)
CAPPRINT(s8)
#endif
# ================================================================= CHECKS (integers only; readings all taken above)
li   t3, 0x1122334455667788
bne  s2, t3, fail_a_src   # A: the untagged source must survive -- THE field defect when it reads 0
bne  t0, t3, fail_a_fwd   # A: rs1 read right after the MOVC
bne  t1, t3, fail_a_dst   # A: rd
bne  s4, x0, fail_b       # B
bne  s5, x0, fail_b
li   t3, 0x55
bne  s6, t3, fail_c       # C
li   t3, 0x77
bne  s7, t3, fail_f       # F: rd = x0 keeps an untagged source
bne  t2, x0, fail_f0      # F: x0 reads 0
bne  s8, x0, fail_g       # G: rs1 = x0 gives cnull

#ifdef BOARD_INTS
  li   a0, 0x600d             # pass marker: every check passed
  CAPPRINT(a0)
#endif
  li   a0, 1
  lla  a1, tohost
  sd   a0, 0(a1)
1:
  j 1b

fail_a_src: li TESTNUM, 1  # NOT_CAP non-zero source nulled -- Q-04 (b) absent
            j fail
fail_a_fwd: li TESTNUM, 2  # rs1 read right after the MOVC returned the wrong value (forwarding)
            j fail
fail_a_dst: li TESTNUM, 3  # rd does not hold the source value
            j fail
fail_b:     li TESTNUM, 4  # NOT_CAP zero case
            j fail
fail_c:     li TESTNUM, 5  # rd == rs1 changed the register
            j fail
fail_f:     li TESTNUM, 6  # rd = x0 nulled an untagged source
            j fail
fail_f0:    li TESTNUM, 7  # x0 does not read 0
            j fail
fail_g:     li TESTNUM, 8  # rs1 = x0 did not give cnull
fail:
#ifdef BOARD_INTS
            CAPPRINT(TESTNUM)  # fail marker: the first failing check's number (1 = Q-04 (b) absent)
#endif
            RVTEST_FAIL

.section .data
.align 6
test_buf_start:
  .zero 256
test_buf_end:

.align 6
.globl tohost
tohost: .dword 0
.globl fromhost
fromhost: .dword 0
