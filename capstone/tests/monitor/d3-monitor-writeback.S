# See LICENSE for license details.

#*****************************************************************************
# d3-monitor-writeback.S
#-----------------------------------------------------------------------------
#
# DOES THE PROPOSED MONITOR FIX SURVIVE LIVE EXCEPTION DELIVERY? D3, validated against the
# R-34/R-24 delivery fix branch, which is the only place it CAN be validated: on the deployed
# bitstream the capability exception is never delivered, so the monitor's current code and its
# replacement both run clean and a pass there means nothing.
#
# THE SITE. capstone-sbi sbi_capstone.S, _handle_non_ecall's emulated-CSR writeback, which runs at
# every rdtime:
#
#     slli t5, t4, 3
#     add  t5, sp, t5          <- integer add: sp is a capability, the result is NOT one
#     sd   a0, 16(t5)          <- plain store through an untagged base
#
# In M-mode with capmode set, the LSU's first clause reads that base as NOT_CAP. An independent
# scan of the 205-line file finds exactly this one site and no other: it is the only memory access
# in it whose base register was last written by integer arithmetic.
#
# THE PROPOSED REPLACEMENT keeps the capability and moves its cursor instead, in place so that
# rd equals rs1 and the source-consumption question does not arise:
#
#     slli t5, t4, 3
#     CINCOFFSET with rd and rs1 both sp, rs2 t5
#     sd   a0, 16(sp)
#     sub  t5, x0, t5
#     CINCOFFSET with rd and rs1 both sp, rs2 t5      <- cursor back where it was
#
# ARMS. Arm B is the replacement and runs FIRST because it is expected to return. Arm A is the
# site as it stands today and runs LAST because on a build without R-24's renumber a live NOT_CAP
# clause enters debug mode instead of trapping, which is a hang, and a hang costs every reading
# that has not been printed yet.
#
#   arm 0  capmode OFF, the monitor's exact shape through an integer base   -> stores, no trap
#          (instrument control: the shape works and the buffer is writable before any gate is on)
#          CAPENTER
#   arm B  the replacement, sp holding a NONLIN RW capability over buf      -> no trap, and the
#          stored value must be READ BACK, because "did not trap" alone is also what a store that
#          never happened looks like; the capability is printed before and after so the restore is
#          on record rather than assumed
#   arm B0 the same replacement at the LOWER EDGE, slot 0 -- the one address this change newly
#          bounds-checks that the preceding SAVE_REG sequence does not already exercise
#   arm A  the site as it stands today, through an integer base             -> cause 24 if the
#          clause is live; a hang if it is live and routed to debug mode
#
# READ THE PAIR. Arm A is the positive control for arm B and the run carries no verdict without
# it: if arm A does not trap, this build does not deliver the exception at all and arm B's clean
# return says nothing about the fix. Arm 0 must store, or the buffer or the shape is wrong and no
# later arm means anything.
#
# EXIT. Every arm after CAPENTER runs with capmode set, and the harness's own pass sequence stores
# to tohost through an auipc-derived integer base -- the same defect, which is why ten tests in the
# branch's sweep time out in their epilogue. This file writes tohost through a capability instead.
#
# This file goes through CPP: no comment line may begin with a hash followed by a directive word,
# and a MACRO(...) form inside a comment would expand.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1                    # TESTNUM must be non-zero or RVTEST_FAIL spins instead of reporting
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s10, 0                   # last observed cause (handler-owned)
  li   s11, 0                   # trap count (handler-owned)

# ---------------------------------------------------------------- arm 0: the shape, capmode OFF
  la   t1, buf
  li   t4, 3                    # the index the monitor computes from mtval
  slli t5, t4, 3
  add  t5, t1, t5               # the monitor's integer add
  li   a0, 0xD3D3
  sd   a0, 16(t5)               # -> buf[5]
  mv   s1, s10                  # cause seen (expect 0)
  li   s10, 0
  la   t1, buf
  ld   s2, 40(t1)               # buf[5] read back (expect 0xD3D3)

# ---------------------------------------------------------------- CAPENTER: capmode sticky from here
  la   a0, _start
  la   a1, _end_of_text
  CAPENTER(a0, a1)

# ---------------------------------------------------------------- gate witnesses
  CAPCREATE(a6)
  li   a0, CAP_TYPE_NONLIN
  CAPTYPE(a6, a0)
  la   a2, buf
  la   a3, buf_end
  CAPBOUND(a6, a2, a3)
  li   a4, CAP_PERM_RW
  CAPPERM(a6, a4)
  CCSRRW(x0, CCSR_CSCRATCH, a6)
  CCSRRW(a7, CCSR_CSCRATCH, x0)
  CAPPRINT(a7)                  # capmode witness: the capability survives the round trip
  csrr t3, mstatus
  CAPPRINT(t3)                  # privilege witness: MPRV bit 17, MPP bits 12:11

# a separate RW capability over buf, kept for the read-backs -- a plain load through an integer
# base would itself be the defect under test
  CAPCREATE(a1)
  li   a0, CAP_TYPE_NONLIN
  CAPTYPE(a1, a0)
  CAPBOUND(a1, a2, a3)
  li   a4, CAP_PERM_RW
  CAPPERM(a1, a4)
  ld   s3, 40(a1)               # buf[5] through a capability, before arm B (expect 0xD3D3)

# ---------------------------------------------------------------- arm B: the replacement
  CAPCREATE(sp)
  li   a0, CAP_TYPE_NONLIN
  CAPTYPE(sp, a0)
  CAPBOUND(sp, a2, a3)
  li   a4, CAP_PERM_RW
  CAPPERM(sp, a4)
  CAPPRINT(sp)                  # the capability before the in-place cursor move
  li   t4, 3
  slli t5, t4, 3
  CINCOFFSET(sp, sp, t5)
  li   a0, 0xB0B0
  sd   a0, 16(sp)               # the replacement's store -> buf[5]
  mv   s4, s10                  # cause seen (expect 0)
  li   s10, 0
  sub  t5, x0, t5
  CINCOFFSET(sp, sp, t5)
  CAPPRINT(sp)                  # after the restore: the cursor must be back where it started
  ld   s5, 40(a1)               # buf[5] after arm B (expect 0xB0B0, or the store never happened)

# ---------------------------------------------------------------- arm B0: the LOWER EDGE, slot 0
# The monitor's store lands at frame_base + slot*8 with slot 0..31, and the SAVE_REG sequence that
# precedes it on every trap writes slots 1..31 through the SAME register -- so slots 1..31 are
# already proved in bounds by code that runs first, and only SLOT 0 is an address this change
# newly subjects to a bounds check that nothing else exercises. It is reachable when the emulated
# instruction's rd is x0. Mimic that geometry exactly: put the cursor 16 BELOW the capability's
# base, as the monitor's is, and store at 16 off it with a zero index, so the access lands on the
# base itself. The cursor is out of bounds in between, which is legal -- the ACCESS is checked,
# not the cursor.
  CAPCREATE(sp)
  li   a0, CAP_TYPE_NONLIN
  CAPTYPE(sp, a0)
  CAPBOUND(sp, a2, a3)
  li   a4, CAP_PERM_RW
  CAPPERM(sp, a4)
  CINCOFFSETIMM(sp, sp, -16)    # the monitor's state at the site: cursor 16 below the frame base
  li   t4, 0                    # rd = x0 -> slot 0
  slli t5, t4, 3
  CINCOFFSET(sp, sp, t5)
  li   a0, 0xE0E0
  sd   a0, 16(sp)               # lands exactly on the capability's base
  mv   s8, s10                  # cause seen (expect 0; 28 would mean the base itself is refused)
  li   s10, 0
  ld   s9, 0(a1)                # buf[0] after arm B0 (expect E0E0, was 4D4F4E49)

  CAPPRINT(s8)                  # arm B0 cause
  CAPPRINT(s9)                  # arm B0 value
  CAPPRINT(s1)                  # arm 0 cause
  CAPPRINT(s2)                  # arm 0 value
  CAPPRINT(s3)                  # buf[5] before arm B
  CAPPRINT(s4)                  # arm B cause
  CAPPRINT(s5)                  # buf[5] after arm B
  CAPPRINT(s11)                 # traps so far

# ---------------------------------------------------------------- arm A: today's site, LAST
  la   t1, buf
  li   t4, 3
  slli t5, t4, 3
  add  t5, t1, t5               # the monitor as it stands today
  li   a0, 0xAAAA
  sd   a0, 16(t5)               # cause 24 if the clause is live; a hang if routed to debug mode
  mv   s6, s10                  # cause seen (expect 24)
  li   s10, 0
  ld   s7, 40(a1)               # buf[5] after arm A: unchanged 0xB0B0 if the store was refused
  CAPPRINT(s6)                  # arm A cause
  CAPPRINT(s7)                  # buf[5] after arm A
  CAPPRINT(s11)                 # total traps

# ---------------------------------------------------------------- exit through a capability
  fence
  CAPCREATE(a5)
  li   a0, CAP_TYPE_NONLIN
  CAPTYPE(a5, a0)
  la   a2, tohost
  addi a3, a2, 8
  CAPBOUND(a5, a2, a3)
  li   a4, CAP_PERM_RW
  CAPPERM(a5, a4)
  li   gp, 1
  sw   gp, 0(a5)
1:
  j    1b

# ---------------------------------------------------------------- trap handler (registers only)
.align 4
trap_handler:
  csrr t0, mcause
  bltz t0, unexpected_trap
  mv   s10, t0
  addi s11, s11, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

unexpected_trap:
  li   TESTNUM, 9
  RVTEST_FAIL

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN

.align 4
buf:
  .dword 0x4D4F4E49             # "MONI"
  .dword 0, 0, 0, 0, 0, 0, 0
buf_end:
  .dword 0x0BADB0B0
  .dword 0

  TEST_DATA

RVTEST_DATA_END
