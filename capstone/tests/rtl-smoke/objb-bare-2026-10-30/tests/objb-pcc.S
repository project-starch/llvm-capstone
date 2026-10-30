# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# objb-pcc.S -- the FETCH face of the lifetime check: does an instruction fetched under a PC capability whose node
#               was just revoked commit? (ObjBounds, 2026-10-30)
#-----------------------------------------------------------------------------
#
# commit_stage.sv's pc_cap_check raises cause 25 when the PC tracker's node is invalid (the tracker adopts the PC
# capability's id as live and is cleared by an invalidation broadcast of its index). CAPSTONE_NO_LIFETIME_CHECK
# removes that arm. In capmode the PC capability's metadata changes only on capability control flow, NOT on a
# trap (frontend.sv npc_metadata), so on bitstream 3 the trap handler runs under the same dead PC capability and
# faults again: the arm does not return there. That is its pre-registered reading, read from the retirement trace.
#
# CONSTRUCTION. The code capability s11 covers [pcarm, _end_of_text) -- the arm and the trap handler -- on its own
# node (the upper piece of a split); s7 is an mrev handle above that node. cjalr enters pcarm under s11.
#   ARM (default): revoke(s7) ends s11's node, so the broadcast names the PC tracker's index.
#   CONTROL (-DPCCTRL): the identical stream, but the revoke ends a DATA piece's node (s8, handle s3), so the PC
#     tracker is untouched. It differs from the arm in one register operand.
# PRE-REGISTERED.
#   e7af48457 / objbounds with the macro undefined:
#     ARM: no print after the revoke; the trace's last retired instruction is at or before the revoke, followed by
#          repeated exception lines at pcarm/pcarm+4 and then at trap_handler, until the cycle limit (FAILED,
#          tohost 0x7fffffff). Bounds, type and permission cannot fire (the control passes all three on the same
#          capability and stream), so the only check left to fire is the lifetime arm.
#     CONTROL: 0x0bb5, 0 (traps), 0 (cause), PASS.
#   ObjBounds (CAPSTONE_NO_LIFETIME_CHECK): ARM and CONTROL both 0x0bb5, 0, 0, PASS.

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
  li   s4, 0
  li   s10, 0

  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)

  CAPCREATE(s6)                  # LIN RWX over the whole text
  li   t1, CAP_TYPE_LIN
  lla  a2, _start
  lla  a3, _end_of_text
  CAPTYPE(s6, t1)
  CAPBOUND(s6, a2, a3)
  li   t1, CAP_PERM_RWX
  CAPPERM(s6, t1)
  lla  t1, pcarm
  SPLIT(s11, s6, t1)             # s11: [pcarm, _end_of_text) on its own node
  MREV(s7, s11)                  # the handle above s11's node

  CAPCREATE(s5)                  # the control's data piece: LIN RW over region, split, its own handle
  li   t1, CAP_TYPE_LIN
  lla  a2, region
  lla  a3, region_end
  CAPTYPE(s5, t1)
  CAPBOUND(s5, a2, a3)
  li   t1, CAP_PERM_RW
  CAPPERM(s5, t1)
  lla  t1, region
  addi t1, t1, 128
  SPLIT(s8, s5, t1)
  MREV(s3, s8)

  lla  t1, pcarm
  SCC(s11, s11, t1)              # cursor at pcarm
  CJALR(t6, s11, 0)              # enter pcarm under s11

.align 6
pcarm:
#ifdef PCCTRL
  REVOKE(s3)                     # CONTROL: a data piece's node; the PC tracker is untouched
#else
  REVOKE(s7)                     # ARM: the PC capability's own node
#endif
  li   a0, 0x0bb5
  CAPPRINT(a0)                   # reached only if fetch after the revoke commits
  CAPPRINT(s10)                  # traps: 0
  CAPPRINT(s4)                   # last cause: 0
  CAP_PASS(s9)

.align 4
trap_handler:
  csrr s4, mcause
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
.align 8
region:
  .zero 256
region_end:
RVTEST_DATA_END
