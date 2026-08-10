# See LICENSE for license details.

#*****************************************************************************
# untagged-ldc-stc-128.S
#-----------------------------------------------------------------------------
#
# Does a 16-byte ldc/stc round trip preserve all 128 bits of PLAIN UNTAGGED DATA?
#
# BOARD OBSERVATION (caplifive_r20.bit, 2026-08-10, inside a capability domain, both
# pointers 16-byte aligned). memcpy's aligned middle loop is exactly
# `ldc a5, 0(src); stc a5, 0(dst)` repeated, and copying 32 bytes of plain data gave:
#
#   src32 = c0c1c2c3c4c5c6c7 c8c9cacbcccdcecf d0d1d2d3d4d5d6d7 d8d9dadbdcdddedf
#   dst32 = c0c1c2c3c4c5c6c7 0000000000000000 d0d1d2d3d4d5d6d7 0000000000000000
#
# i.e. every 16-byte chunk keeps its LOW 8 bytes and gets ZEROES in its HIGH 8. That copy
# path exists precisely so a struct containing pointers keeps its capability TAGS; it is
# only correct if untagged ldc/stc is also bit-exact over the full 128-bit word. QEMU was
# fixed for this (capstone-qemu fix/untagged-ldc-stc-128bit-preservation), so QEMU cannot
# answer the question and every SQLite result taken under QEMU is blind to it.
#
# WHY SIMULATION. On the board this was inferred from bytes read back through a whole
# memcpy. Here the RVFI trace prints the value each `ld` retires, so the two halves are read
# DIRECTLY, with the plain-store control in the same run.
#
# WHAT TO LOOK FOR in the RVFI trace, on the four `ld` instructions that write t3/t4/t5/t6:
#   t3 = 0123456789abcdef   low half survived the ldc/stc round trip
#   t4 = fedcba9876543210   HIGH half survived        -> no defect
#   t4 = 0000000000000000   HIGH half LOST            -> reproduces the board
#   t5/t6 must equal the same two constants: they are the CONTROL, written and read with
#   plain sd/ld to a different offset, never touching ldc/stc.
#
# If t5/t6 are wrong, the buffer or the capability is at fault and the run carries NO verdict
# about ldc/stc. If t3/t4 and t5/t6 are all correct, the defect does NOT reproduce here and
# the board result has some other cause (domain mapping, D-cache, or the copy loop itself).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc          # match the board's 4-byte encodings; the assembler otherwise
                       # emits compressed forms and compression becomes a hidden variable
.section .text.init
.globl _start
_start:

# NOTES, each of which cost a run elsewhere in this directory:
#  * CHK_START / CHK_END read a0/a1 as plain integers and trap when those hold live
#    capabilities. Not used here.
#  * A LINEAR capability MOVES rather than copies, so one parked capability cannot serve two
#    arms. Two are built up front.
#  * Helper registers must never have held a capability: an ordinary instruction does not
#    clear a register's capability metadata, and a stale tag turns a plain-integer operand
#    into UNEXPECTED_OPERAND. a0/a2/a3/a4 are helpers ONLY.
#  * This file goes through the C preprocessor, so a MACRO(...) form inside a comment
#    expands. Written as "CAPTYPE with CAP_TYPE_LIN" rather than the call form.

#define MKCAP(CREG)          \
  CAPCREATE(CREG);           \
  li  a0, CAP_TYPE_LIN;      \
  lla a2, test_buf_start;    \
  lla a3, test_buf_end;      \
  li  a4, CAP_PERM_RWX;      \
  CAPTYPE(CREG, a0);         \
  CAPBOUND(CREG, a2, a3);    \
  CAPPERM(CREG, a4);

MKCAP(a5)
MKCAP(a6)

# ---------------------------------------------------------------- lay down the pattern
# Two plain 64-bit stores build a 16-byte word of ORDINARY DATA at offset 0. Nothing here
# creates a capability, so the line's tag is 0 and `ldc` below loads untagged data.
MOVC(a1, a5)
li   t0, 0x0123456789abcdef
li   t1, 0xfedcba9876543210
sd   t0, 0(a1)
sd   t1, 8(a1)

# ---------------------------------------------------------------- ARM A: ldc/stc round trip
# Load the 16 bytes as one capability-grained word and store them back at offset 32, exactly
# as memcpy's middle loop does, then read both halves with plain loads.
LDC(a7, a1, 0)          # ldc a7, 0(a1)   -- a7 now holds 128 bits of untagged data
STC(a1, a7, 32)         # stc a7, 32(a1)
ld   t3, 32(a1)         # LOW  half  -- RVFI prints what was actually stored
ld   t4, 40(a1)         # HIGH half  -- 0 here reproduces the board

# ---------------------------------------------------------------- ARM B: CONTROL, plain path
# The same two 64-bit values, written and read back with sd/ld only, at offset 64. This never
# touches ldc/stc, so it validates the buffer, the bounds and the capability itself. If this
# arm is wrong the run proves nothing.
MOVC(a1, a6)
sd   t0, 64(a1)
sd   t1, 72(a1)
ld   t5, 64(a1)         # CONTROL low  -- must be 0123456789abcdef
ld   t6, 72(a1)         # CONTROL high -- must be fedcba9876543210

# ---------------------------------------------------------------- done
  li   a0, 1
  lla  a1, tohost
  sd   a0, 0(a1)
1:
  j 1b

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
