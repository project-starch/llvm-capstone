# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# s1-revoke-gap.S  -- does a revoke pay for an mrev that retired long before it?  (2026-10-14)
#-----------------------------------------------------------------------------
#
# WHY. capstone-sublet step 6 (d93a8d66b) reorders mrev (check, deepen-write, handle). In simulation the first mrev
# of a program then retires 8 cycles sooner and the timed revoke after it commits at the SAME absolute cycle as
# before (r12-s1-n8: 4284 on both trees), so the revoke's mcycle bracket reads +8 while the program's total time is
# unchanged. The revoke in that arm was ~440 cycles after the last mrev with only lcc instructions between. What pins
# the revoke's completion is unresolved: a drain of mrev's last node write that later instructions hide, or an
# artefact of the simulation memory model. The board's R1 harness brackets a revoke whose handle was minted at fixture
# time (ldc handle; csrr; revoke; csrr), so the two readings disagree about silicon: this arm separates them with ONE
# binary that runs on both the FP and the Sublet bitstream (legacy instructions only).
#
# SHAPE. a0 is the linear base (capenter, node 2). A capcreate'd capability carries the base's node, and the first
# revoke below a handle on the base kills that node, so a park capability made that way, the final tohost store and
# the board wrapper's record capability all die at k = 1 (the first draft did exactly that: ldc trapped 25 at
# k = 2..4 on every tree; alloc off the sentinel id traps 24 on these trees). So the first mint is an mrev handle M0
# on the base that is never revoked: it sits shallowest and first in the list, outside every walk the timed revokes
# make. The park capability (s6, NONLIN RW over `park`) takes M0's node with capnode, which copies the node id from a
# capability register (r12-recl-stale's U10 transplant); M0 stays in s2 for the whole run and the board wrapper's
# record macros copy from it too (BOARD_REC_NODE = x18). If the node were wrong the first stc would trap and the
# trap count would say so. Four handles are then minted on the base, outermost first, and parked through s6 (stc). Then,
# innermost first, each repeat: [GAP iterations of a plain ALU loop, timed, no memory and no capability instruction],
# ldc the handle, csrr, revoke, csrr. Revoking the innermost kills the base; each later revoke walks the run of
# corpses below it, so the per-index cost differs by design and is compared per index across trees and bitstreams.
# Only k = 1 follows an mrev; k = 2..4 follow the previous revoke and are same-binary repeat controls.
#   GAP = 0     the S1 shape: the first revoke sits a few instructions behind the last mrev
#   GAP = 1000  the R1 shape: >= 2,000 cycles of ALU work between the last mrev (or the previous revoke) and the timed
#               revoke, so a memory drain cannot hide behind loads
#
# READINGS, per repeat k = 1..4: gap cycles (the loop's own bracket; ~8 for GAP 0), revoke cycles, revoke cause
# (expect 0). Then: a0's validity after the first revoke (expect 0), total traps (expect 0), GAP, 4. Sixteen values.
# PREDICTIONS (per tree, filled from the simulation before the board run, in the RTL plan doc):
#   outcome A: both arms' revoke cycles identical on the step-5 and step-6 trees at every k -> the pin is mrev-adjacent
#              or a model artefact, and the FP/Sublet silicon readings should match;
#   outcome B: k = 1 differs by 8 on GAP 0 and not on GAP 1000 -> the pin is a drain shorter than the gap;
#   outcome C: k = 1 differs by 8 on BOTH arms -> the pin is not a drain of mrev's write; R1's intercept moves on silicon.
# Registers x8, x16, x17 (the board wrapper's record scratch) are never named here; x23..x31 are free until the final
# report, which is the only place the wrapper uses them.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#ifndef GAP
#define GAP 0
#endif

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s10, 0
  li   s11, 0

  CAPENTER(x0, x0)              # a0 = linear data cap (the base)

# ---------------------------------------------------------------- M0 (s2), never revoked; s6: NONLIN RW over `park` on M0's node
  MREV(s2, a0)
  CAPCREATE(s6)
  li   t0, CAP_TYPE_NONLIN
  CAPTYPE(s6, t0)
  CAPNODE(s6, s2)
  lla  t1, park
  lla  t2, park_end
  CAPBOUND(s6, t1, t2)
  li   t0, CAP_PERM_RW
  CAPPERM(s6, t0)

# ---------------------------------------------------------------- four handles on the base, outermost first, parked at 16*(4-k)
  MREV(a1, a0)                  # k = 4 (outermost, revoked last)
  STC(s6, a1, 0)
  MREV(a2, a0)                  # k = 3
  STC(s6, a2, 16)
  MREV(a3, a0)                  # k = 2
  STC(s6, a3, 32)
  MREV(a4, a0)                  # k = 1 (innermost, revoked first; its walk kills the base)
  STC(s6, a4, 48)

# ---------------------------------------------------------------- the repeats
#define REPEAT(OFF) \
  li   t2, GAP; \
  csrr t3, mcycle; \
  61: beqz t2, 62f; addi t2, t2, -1; j 61b; \
  62: csrr t4, mcycle; \
  sub  s1, t4, t3; \
  LDC(t1, s6, OFF); \
  csrr t3, mcycle; \
  REVOKE(t1); \
  csrr t4, mcycle; \
  sub  s9, t4, t3; \
  CAPPRINT(s1); \
  CAPPRINT(s9); \
  CAPPRINT(s10); li s10, 0

  REPEAT(48)
  LCC(s7, a0, x0)               # the base after the first revoke: expect 0
  REPEAT(32)
  REPEAT(16)
  REPEAT(0)

# ---------------------------------------------------------------- report
  CAPPRINT(s7)                  # base validity after revoke 1: 0 = dead
  CAPPRINT(s11)                 # total traps, expect 0
  li   t6, GAP
  CAPPRINT(t6)
  li   t6, 4
  CAPPRINT(t6)
#ifdef BOARD_REC_NODE
  CAP_PASS(s11)                 # the board wrapper's node-aware report (BOARD_REC_NODE = s2 = x18)
#else
  CAPCREATE(s8)                 # the tohost store through a capability on the surviving node
  li   t0, CAP_TYPE_LIN
  CAPTYPE(s8, t0)
  CAPNODE(s8, s2)
  lla  t1, tohost
  lla  t2, tohost + 64
  CAPBOUND(s8, t1, t2)
  li   t0, CAP_PERM_RWX
  CAPPERM(s8, t0)
  fence
  li   t0, 1
  sw   t0, 0(s8)
  1: j 1b
#endif

.align 4
trap_handler:
  csrr t0, mcause
  mv   s10, t0
  addi s11, s11, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 6
park:
  .zero 128
park_end:
RVTEST_DATA_END
