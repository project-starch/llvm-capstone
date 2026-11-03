# lv-t1-body.h -- conformance rule T1: control transfer through a code capability whose lifetime has ended.
# This file goes through cpp: instruction names are written in lower case in comments.
#
# ISA. A capability whose node has been revoked is invalid (cap-man-insn.adoc, revoke: c.valid is set to 0 for
# every capability aliasing the revoked node). cjalr itself checks only the tag (ctrl-flow-insn.adoc), so the
# refusal belongs to the fetch: prog-model.adoc says an instruction fetch raises an exception when pc.valid is 0,
# so no instruction at the target may execute. The cause the academic spec at the superproject pin names for the
# fetch is 1 (instruction access fault); the project's ruled code for a dead PCC is 25 (isa.md "PCC must have
# ... a live node", ISSUES Q-29: "dead 25"), which is what commit_stage raises. Either cause is a PASS; any
# retired instruction at the target is a FAIL.
#
# SHAPE. M mode, capmode set by capenter (PCC = the capenter code capability, node 1). A live NONLIN RW data
# capability s1 covers the marker word. A code capability C (s4) covers [tgt, tgt_end) on its OWN node. C's
# lifetime is ended with the revision's own instructions while the hart runs under the capenter PCC:
#   bitstream 3 (asm_insn.h defines cderive): C = cderive child of a NONLIN MANAGE parent; crevoke ends it.
#   HEAD (no cderive): C = the split-off piece of a LINEAR region, an mrev handle above it, delin to NONLIN,
#   revoke of the handle ends the piece's node.
# The revoking instruction returns only after the rev-node unit has written the record; a fence and the lcc
# validity query (reading 0) confirm completion before the jump. Then cjalr ra, C, 0. The target's FIRST
# instruction stores MARK to the marker word, the second returns through s6 (a live return capability).
#
# LV_OOB (added after the first readings, PC-check control): live C, cjalr with imm 16 so the target is tgt_end,
# outside C's bounds. Shows whether the commit-stage PC check fires at all on the revision (28 expected). The trap
# pcs and the handler's retired mcause reads are in the RVFI trace.
# ARMS (one define each): LV_FIRE (revoke, then jump), LV_NOREVOKE (control a: identical, no revoke),
# LV_UNTAGGED (control b: identical to LV_NOREVOKE but the jump goes through an untagged register holding tgt).
#
# READINGS (CAPPRINT, in order):
#   R1  0x7101                 opening marker
#   R2  C, the code capability (full print: id, type, perm, bounds)
#   R3  lcc validity of C before the revoke step: 1
#   R4  lcc validity of C after the revoke step: FIRE 0, others 1
#   R5  the marker word before the jump: 0
#   R6  the marker word after the jump returned: MARK if the target ran, 0 if it did not
#   R7  trap count            R8 last mcause      R9 last mepc minus tgt      R10 last mtval
#   R11 0x59                   closing marker
# FIRE conforming: the target's first fetch traps; no instruction can commit afterwards (a trap keeps the PCC
# metadata, frontend.sv), so the run ends at the time-out with a trap at tgt and no retired store to the marker
# word in the RVFI trace. FIRE deviating: R6 = MARK, trap count 0, the run ends SUCCESS.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#ifndef CAP_PASS
#define CAP_PASS(creg) \
    CAPCREATE(creg); li a0, CAP_TYPE_LIN; lla a2, tohost; lla a3, tohost + 64; li a4, CAP_PERM_RWX; \
    CAPTYPE(creg, a0); CAPBOUND(creg, a2, a3); CAPPERM(creg, a4); \
    fence; li a0, 1; sw a0, 0(creg); 1: j 1b
#endif

#define MARK 0x4d41524b

#define NODEQ(CAP) \
  LCC(t6, CAP, x0); \
  CAPPRINT(t6)

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s10, 0                   # trap count
  li   s9, 0                    # last mcause
  li   s7, 0                    # last mepc
  li   s8, 0                    # last mtval

  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)

  li   t0, 0x7101
  CAPPRINT(t0)                  # R1

# ---------------------------------------------------------------- s1: live NONLIN RW data capability, the marker word
  CAPCREATE(s1)
  li   t0, CAP_TYPE_NONLIN
  lla  t1, mark
  lla  t2, mark_end
  CAPTYPE(s1, t0)
  CAPBOUND(s1, t1, t2)
  li   t0, CAP_PERM_RW
  CAPPERM(s1, t0)
  sd   x0, 0(s1)

# ---------------------------------------------------------------- C: the code capability on its own node
#ifdef CDERIVE
  CAPCREATE(s2)                 # parent: NONLIN, MANAGE plus RWX, over [tgt, tgt_end), cursor tgt
  li   t0, CAP_TYPE_NONLIN
  lla  t1, tgt
  lla  t2, tgt_end
  CAPTYPE(s2, t0)
  CAPBOUND(s2, t1, t2)
  li   t0, CAP_PERM_ALL
  CAPPERM(s2, t0)
  lla  t1, tgt
  lla  t2, tgt_end
  sub  t0, t2, t1
  CDERIVE(s4, s2, t0)           # C: NONLIN child over [tgt, tgt_end), perm RWX plus MANAGE, a fresh node
#else
  CAPCREATE(s2)                 # a LINEAR RWX region over [tgt_pre, tgt_end)
  li   t0, CAP_TYPE_LIN
  lla  t1, tgt_pre
  lla  t2, tgt_end
  CAPTYPE(s2, t0)
  CAPBOUND(s2, t1, t2)
  li   t0, CAP_PERM_RWX
  CAPPERM(s2, t0)
  lla  t1, tgt
  SPLIT(s4, s2, t1)             # C: [tgt, tgt_end) on a fresh node, cursor tgt
  MREV(s3, s4)                  # the handle, a node above C's
  DELIN(s4)                     # C becomes NONLIN, same node
#endif
  CAPPRINT(s4)                  # R2
  NODEQ(s4)                     # R3: 1

# ---------------------------------------------------------------- end C's lifetime, under the capenter PCC
#ifdef LV_FIRE
#ifdef CDERIVE
  CREVOKE(s2, s4)
#else
  REVOKE(s3)
#endif
#endif
  fence
  nop
  nop
  nop
  nop
  NODEQ(s4)                     # R4: FIRE 0 (revocation complete), others 1

# ---------------------------------------------------------------- the return capability
# The cjalr link is an integer, pc+4 (read on HEAD, same flu code on bitstream 3), so the target returns through s6 instead: a live NONLIN
# RX capability over the caller text with its cursor at after_jump, on the capcreate node.
  CAPCREATE(s6)
  li   t0, CAP_TYPE_NONLIN
  lla  t1, _start
  lla  t3, _end_of_text
  CAPTYPE(s6, t0)
  CAPBOUND(s6, t1, t3)
  li   t0, CAP_PERM_RX
  CAPPERM(s6, t0)
  lla  t1, after_jump
  lla  t3, _start
  sub  t1, t1, t3
  CINCOFFSET(s6, s6, t1)

# ---------------------------------------------------------------- the jump
  li   t2, MARK
  ld   t3, 0(s1)
  CAPPRINT(t3)                  # R5: 0
#ifdef LV_UNTAGGED
  lla  t4, tgt                  # an integer: no tag
  CJALR(ra, t4, 0)
#elif defined(LV_OOB)
  CJALR(ra, s4, 16)             # C's cursor moved to tgt_end: outside C's bounds
#else
  CJALR(ra, s4, 0)
#endif
after_jump:
  ld   t3, 0(s1)
  CAPPRINT(t3)                  # R6: MARK if the target ran
  CAPPRINT(s10)                 # R7
  CAPPRINT(s9)                  # R8
  lla  t1, tgt
  sub  t1, s7, t1
  CAPPRINT(t1)                  # R9
  CAPPRINT(s8)                  # R10
  li   t0, 0x59
  CAPPRINT(t0)                  # R11
  CAP_PASS(s11)

.align 4
trap_handler:
  csrr s9, mcause
  csrr s7, mepc
  csrr s8, mtval
  addi s10, s10, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

# ---------------------------------------------------------------- the target routine
.align 4
tgt_pre:
  nop
  nop
  nop
  nop
tgt:
  sd   t2, 0(s1)                # the marker store: the first instruction under C
  CJALR(x0, s6, 0)              # back to after_jump under the return capability
  nop
  nop
tgt_end:
#ifdef LV_OOB
  sd   t2, 8(s1)                # the out-of-bounds landing: writes MARK to the second word if it retires
  CJALR(x0, s6, 0)
#endif

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 6
mark:
  .zero 64
mark_end:
  TEST_DATA
RVTEST_DATA_END
