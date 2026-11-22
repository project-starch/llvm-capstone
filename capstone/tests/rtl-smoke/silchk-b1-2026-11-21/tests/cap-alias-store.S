# cap-alias-store.S -- item 4 / R-60, ACCEPTANCE READING 4: an out-of-window cursor must not ALIAS.
#
# THE HAZARD. A load or store through a capability whose cursor lies outside its representable window is
# bounds-checked against the bounds DECODED AT THAT CURSOR -- the true bounds shifted by whole window spans --
# so when the shifted decode contains the cursor the access PASSES and touches memory a span away with no trap.
# For this geometry the span is 0x4000 (2^(E+14) with E = 0), and a decode at a cursor always lands in the
# 0x4000-aligned block containing that cursor, so the shifted decode CONTAINS the cursor by construction.
# `region` below is .align 14, i.e. 0x4000-aligned, so region+0x4000 is both OUT of window and an address the
# shifted bounds admit. The triggering condition is therefore CREATED here, not assumed -- which is the whole
# difference between this arm and a directed test that comes back clean having tested nothing.
#
# WHY THIS ARM HAS ITS OWN .data, where cap-cursor-window uses an absolute GRP_BASE: that arm says of its own
# mint "the capability is never dereferenced", and its address is the runtime's rather than this image's. Reading
# 4 is the one reading that MUST dereference, so every address here is backed by this image's own .data.
#
# THE READING IS TWO-SIDED, which is what makes it more than a trap count. A sentinel is written a span away
# through an INDEPENDENT, in-window capability and read back through that same capability afterwards:
#   before the fix   the store LANDS      -> the sentinel is OVERWRITTEN and nothing traps
#   after the fix    the tag is CLEARED   -> the sentinel is INTACT and the store traps
# A trap count alone cannot tell "refused" from "wrote somewhere else", and the sentinel is what separates them.
#
# THE CAUSE IS 24, AND THE REASON MATTERS AS MUCH AS THE NUMBER. The batch plan pre-registers 28 for this
# reading, and 28 IS the bounds fault on this RTL -- `enum ex_code` is mcause = 23 + ordinal, so
# UNEXPECTED_OPERAND 24, INVALID_CAPABILITY 25, UNEXPECTED_CAP_TYPE 26, INSUFFICIENT_PERMISSION 27,
# OUT_OF_BOUNDS 28, ILLEGAL_OPERAND_VALUE 29, and commit_stage.sv:388 writes `64'd28; // OUT_OF_BOUNDS
# (23 + 5)` in as many words. So 28 was the PRE-FIX reading of this hazard: QEMU, and the board's
# bounds-malloc control on silicon, trap a bounds violation AT THE ACCESS.
# After item 4 the fault moves EARLIER and therefore changes code. The out-of-window CINCOFFSET has already
# cleared the tag by the time the store issues, so the store goes through a NON-CAPABILITY and the cause is
# 24, UNEXPECTED_OPERAND -- not a bounds fault at all, because no bounds check is ever reached. That is the
# pre-registration here. Grading this arm against 28 would read a correct result as a failure, and reading 24
# as "the bounds check moved" would be the same mistake inverted.

# Predictions, written before the run:
#   A  0x5a5a  the sentinel is readable through the witness   (instrument control; if not, the arm is VOID)
#   B  0x1234  an IN-WINDOW store/load through the test cap still works (control; if not, the arm is VOID)
#   C  trap count before the aliasing store
#   D  trap count after:   C+1 after the fix, C before it
#   E  cause 24 (UNEXPECTED_OPERAND), the store meeting a non-capability -- NOT 28, no bounds check is reached
#   F  0x5a5a  THE READING: the sentinel INTACT after the fix; 0x9999 before it, the alias having landed
#   G  7       the moved capability's LCC type query: NOT_CAP (the selector returns type - 1 in three bits)

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

  .option norvc

  .section .text.init
  .globl _start
_start:

  li   gp, 1
  li   s10, 0
  li   s11, 0
  la   t0, m_trap_handler
  csrw mtvec, t0

  CAPENTER(a0, a1)

  # ---- the WITNESS: an independent, in-window capability over the page a span away -------------
  li   gp, 2
  CAPCREATE(s1)
  li   t0, CAP_TYPE_NONLIN
  lla  a2, alias_target
  lla  a3, alias_target_end
  li   t1, CAP_PERM_RW
  CAPTYPE(s1, t0)
  CAPBOUND(s1, a2, a3)
  CAPPERM(s1, t1)

  li   t2, 0x5a5a
  sw   t2, 0(s1)                  # the sentinel, a span away from the region under test
  lw   t2, 0(s1)
  CAPPRINT(t2)                    # A: 0x5a5a -- the witness works at all

  # ---- the capability UNDER TEST over the region --------------------------------------------------
  li   gp, 3
  CAPCREATE(s2)
  li   t0, CAP_TYPE_NONLIN
  lla  a2, region
  lla  a3, region_end
  li   t1, CAP_PERM_RW
  CAPTYPE(s2, t0)
  CAPBOUND(s2, a2, a3)
  CAPPERM(s2, t1)

  # ---- CONTROL FIRST, in window: this must keep working, or the arm says nothing ------------------
  li   gp, 4
  li   t6, 0x800
  CINCOFFSET(s4, s2, t6)
  li   t2, 0x1234
  sw   t2, 0(s4)
  lw   t2, 0(s4)
  CAPPRINT(t2)                    # B: 0x1234 -- an in-window store/load is untouched by item 4

  # ---- THE ALIASING STORE: one span out, where the shifted decode admits the access ---------------
  li   gp, 5
  CAPPRINT(s11)                   # C: the trap count before
  li   t6, 0x4000
  CINCOFFSET(s3, s2, t6)          # OUT of window: item 4 must clear the tag here
  li   t2, 0x9999
  sw   t2, 0(s3)                  # before the fix this LANDS on alias_target; after it, it traps
  CAPPRINT(s11)                   # D: the trap count after -- C+1 after the fix
  CAPPRINT(s10)                   # E: the cause -- 24, a store through a non-capability

  # ---- THE READING: was memory a span away touched? ----------------------------------------------
  li   gp, 6
  lw   t2, 0(s1)                  # read through the WITNESS, which was never moved
  CAPPRINT(t2)                    # F: 0x5a5a intact (fixed) or 0x9999 (the alias landed)

  li   gp, 7
  LCC(t1, s3, x1)                 # the type query: selector 1 returns cap_type - 1 in three bits
  CAPPRINT(t1)                    # G: 7 = NOT_CAP, the tag cleared

  li   t0, 0x58
  CAPPRINT(t0)
  j    pass

  .align 4
m_trap_handler:
  csrr t0, mcause
  mv   s10, t0
  addi s11, s11, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

pass:
  CAP_PASS(a7)                    # NOT RVTEST_PASS: capmode refuses its integer-base store to tohost

  RVTEST_CODE_END
_end_of_text:

  .data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END

  # 0x4000-aligned so the window geometry is exact: region+0x4000 is the first address outside the
  # representable window AND the base of the shifted decode, hence an address that decode admits.
  .align 14
region:
  .fill 0x1000, 1, 0
region_end:
  .fill 0x3000, 1, 0              # filler: carries region_end up to region + 0x4000
alias_target:
  .fill 0x1000, 1, 0
alias_target_end:
