# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# stale-access.S  -- what a data LOAD and a data STORE through a REVOKED capability raise on the M-mode LSU path (2026-10-16)
#-----------------------------------------------------------------------------
#
# For the bug corpus on the FPGA (docs/plans/2026-10-15-bug-corpus-on-fpga.md, ask 1): every temporal verdict depends
# on whether a data access through a capability whose node has been revoked is refused on this RTL (R-35, archived:
# fixed) or retires (the plain-temporal case.json disclaimer, citing Q-11). Three revocation paths, each followed by
# a load and a store through the stale capability, the trap handler recording mcause and resuming after the faulting
# instruction. Readings, in order (CAPPRINT values):
#   path A, crevoke: a NONLIN child of the base (cderive), a value stored and loaded back through it (0x77), then
#     crevoke(base, child); #1 ld through the child -> mcause; #2 sd through the child -> mcause
#   path B, revoke of a handle: a LINEAR region on a node of its own (the upper piece of a split: a capcreate'd
#     capability carries the base's node, and a revoke above THAT would end the base, the park and everything else, as
#     the first draft of this arm did), an mrev handle above it, revoke(handle); the piece's capability is now on a
#     dead node: #3 ld -> mcause; #4 sd -> mcause
#   path C, the stale capability reloaded from memory: the child of path A was parked with stc BEFORE the crevoke and
#     is reloaded with ldc after it: #5 the ldc's mcause (0 if the load itself is allowed), #6 ld through the reloaded
#     capability -> mcause; #7 sd -> mcause
#   #8 the value the base still reads at the child's offset (0x77 if neither stale store landed, the stored value if
#     one did: the datum that decides "retires" against "refused")
#   #9 the trap count (6: exactly the six probed accesses trapped, none else)
# Predicted on d93a8d66b, from R-45's flush and the LSU's node check: 25 for every stale load and store (#1 #2 #3 #4
# #6 #7), 0 at #5 (ldc of a capability word is a plain capability load; the node is checked at use), 0x77 at #8, 6 at
# #9. Any access that reads 0 at its mcause position retired, and #8 then shows whether the store landed.
# Registers x8, x16, x17 are left to the board wrapper; this arm never revokes the base's node (the base is NONLIN
# with its children ended by crevoke in A; B's handle sits above a split piece's own node, not above the base).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s9, 0                    # last mcause
  li   s10, 0                   # traps

  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)

# ---------------------------------------------------------------- the base: NONLIN RW + MANAGE over `data`
  CAPCREATE(s2)
  li   t0, CAP_TYPE_NONLIN
  lla  t1, data
  lla  t2, data_end
  CAPTYPE(s2, t0)
  CAPBOUND(s2, t1, t2)
  li   t0, CAP_PERM_ALL
  CAPPERM(s2, t0)
  sd   x0, 0(s2)
  sd   x0, 64(s2)

# ---------------------------------------------------------------- the park: NONLIN RW over `park`, on its own cderive child of the base
  li   t0, 256
  CDERIVE(s3, s2, t0)           # the park lives at data[0, 256): its node is a child never revoked

# ---------------------------------------------------------------- path A: cderive a child at data+512, use it, park it, crevoke it
  lla  t1, data
  li   t0, 512
  add  t1, t1, t0
  CINCOFFSET(s4, s2, t0)        # cursor at data+512
  li   t0, 64
  CDERIVE(s5, s4, t0)           # the child over [data+512, data+576)
  li   t2, 0x77
  sd   t2, 0(s5)
  ld   t3, 0(s5)
  STC(s3, s5, 0)                # parked for path C (a NONLIN capability: stc keeps the register)
  CREVOKE(s2, s5)               # the child is dead
  li   s9, 0
  ld   t3, 0(s5)                # #1
  CAPPRINT(s9)
  li   s9, 0
  li   t2, 0x1111
  sd   t2, 0(s5)                # #2
  CAPPRINT(s9)

# ---------------------------------------------------------------- path B: a LINEAR region, an mrev handle, revoke
  CAPCREATE(s6)
  li   t0, CAP_TYPE_LIN
  lla  t1, region
  lla  t2, region_end
  CAPTYPE(s6, t0)
  CAPBOUND(s6, t1, t2)
  li   t0, CAP_PERM_RW
  CAPPERM(s6, t0)
  sd   x0, 0(s6)
  lla  t1, region
  li   t0, 128
  add  t1, t1, t0
  SPLIT(s11, s6, t1)            # s11: [region+128, region_end) on its own node (the split sibling)
  sd   x0, 0(s11)
  MREV(s7, s11)                 # the handle, above the piece's node
  REVOKE(s7)                    # ends the piece's node and nothing else
  li   s9, 0
  ld   t3, 0(s11)               # #3
  CAPPRINT(s9)
  li   s9, 0
  li   t2, 0x2222
  sd   t2, 0(s11)               # #4
  CAPPRINT(s9)

# ---------------------------------------------------------------- path C: the parked child, reloaded after its crevoke
  li   s9, 0
  LDC(s8, s3, 0)                # #5
  CAPPRINT(s9)
  li   s9, 0
  ld   t3, 0(s8)                # #6
  CAPPRINT(s9)
  li   s9, 0
  li   t2, 0x3333
  sd   t2, 0(s8)                # #7
  CAPPRINT(s9)

# ---------------------------------------------------------------- the datum
  li   t0, 512
  CINCOFFSET(s4, s2, t0)
  ld   t3, 0(s4)                # #8: 0x77 if no stale store landed
  CAPPRINT(t3)
  CAPPRINT(s10)                 # #9: the trap count
  CAP_PASS(s11)

.align 4
trap_handler:
  csrr s9, mcause
  addi s10, s10, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 6
data:
  .zero 1024
data_end:
.align 6
park:
  .zero 64
park_end:
.align 6
region:
  .zero 256
region_end:
RVTEST_DATA_END
