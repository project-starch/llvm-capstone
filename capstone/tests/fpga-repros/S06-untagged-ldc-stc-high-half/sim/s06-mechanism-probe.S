# See LICENSE for license details.

#*****************************************************************************
# s06-mechanism-probe.S
#-----------------------------------------------------------------------------
#
# Four questions about S-06 that must be answered BEFORE any RTL is written, batched into one
# image so they cost one simulation rather than four.
#
# S-06: an ldc/stc round trip over PLAIN UNTAGGED 16-byte data keeps only the LOW 8 bytes. The
# mechanism is two gates in core/cache_subsystem/wt_dcache_mem.sv (line numbers on THIS branch,
# fpga-testing-dev-s06): the load forces the high half to zero when the line's shadow tag is
# clear (:310), and the store then requests only the bank matching its own offset because
# st_wr_cap = |wr_user_i is 0 (:140, :230).
#
# READ THE ARMS IN ORDER. Arm A is the INSTRUMENT CONTROL. If the plain sd/ld path does not
# return both constants, the buffer or the capability is at fault and NO later arm carries a
# verdict -- a clean result downstream would then be a fact about the instrument.
#
#   arm A  plain sd/ld, no ldc/stc         CONTROL, both halves must survive
#   arm B  ldc/stc, NON-ZERO high half     baseline: does S-06 reproduce on THIS branch?
#   arm C  ldc/stc, ZERO high half, over a destination whose high half is NON-ZERO
#   arm D  what cap_type does an ordinary data pointer actually carry?
#
# WHY ARM C EXISTS, and it is the one that is genuinely new. Both width signals in the cache are
# gated on CONTENT (|user), not on the opcode. A 16-byte chunk whose high 8 bytes are exactly
# zero therefore degenerates to a single-bank write. If that is right, a copy of such a chunk
# FAILS TO OVERWRITE a destination that already holds a non-zero high half -- i.e. S-06 for one
# of the commonest shapes there is, a struct with a zero field. No arm in the S-06 package
# creates that condition, so it has never been tested. Arm C sets the destination to a poison
# value first, so a stale-destination result is unmistakable rather than looking like a zero.
#
# WHY ARM D EXISTS. The proposed fix has a "tagged" path whose behaviour differs for NONLIN
# versus LINEAR capabilities (ldc clears the source memory only for non-NONLIN). If ordinary
# compiled data pointers are always NONLIN -- which is what the entry glue's delin would produce
# -- then the clearing path is never exercised by real code, and building or "validating" it
# would prove nothing. This arm reproduces the glue's own idiom rather than asserting the answer
# from source reading.
#
# This file goes through the C preprocessor, so a MACRO(...) form inside a comment expands.
# Instruction names in comments are written bare for that reason.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc          # keep 4-byte encodings; compression is otherwise a hidden variable

.section .text.init
.globl _start
_start:

# Helper registers a0/a2/a3/a4 must NEVER hold a capability: an ordinary instruction does not
# clear a register's capability metadata, and a stale tag turns a plain-integer operand into
# UNEXPECTED_OPERAND.
#define MKCAP(CREG, CTYPE)   \
  CAPCREATE(CREG);           \
  li  a0, CTYPE;             \
  lla a2, test_buf_start;    \
  lla a3, test_buf_end;      \
  li  a4, CAP_PERM_RWX;      \
  CAPTYPE(CREG, a0);         \
  CAPBOUND(CREG, a2, a3);    \
  CAPPERM(CREG, a4);

# NONLIN so no arm consumes the address capability itself.
MKCAP(a1, CAP_TYPE_NONLIN)

li   t0, 0x0123456789abcdef      # low  constant
li   t1, 0xfedcba9876543210      # high constant

# ---------------------------------------------------------------- ARM A: INSTRUMENT CONTROL
# The same two constants, written and read with plain sd/ld only, never touching ldc/stc.
sd   t0, 0(a1)
sd   t1, 8(a1)
ld   s2, 0(a1)          # ARM A low  -- must be 0123456789abcdef
ld   s3, 8(a1)          # ARM A high -- must be fedcba9876543210

# ---------------------------------------------------------------- ARM B: S-06 baseline
# Lay a 16-byte plain pattern at 64, round-trip it through ldc/stc to 96, read both halves.
sd   t0, 64(a1)
sd   t1, 72(a1)
LDC(a5, a1, 64)         # ldc a5, 64(a1)  -- 128 bits of untagged data
STC(a1, a5, 96)         # stc a5, 96(a1)
ld   s4, 96(a1)         # ARM B low  -- expect 0123456789abcdef
ld   s5, 104(a1)        # ARM B high -- 0 reproduces S-06 on this branch

# ---------------------------------------------------------------- ARM C: THE NEW QUESTION
# Destination at 160 is POISONED with a non-zero high half first, so "stale" is distinguishable
# from "zero". Source at 128 has a non-zero low half and a ZERO high half.
li   t2, 0xdeadbeefcafef00d
sd   t2, 160(a1)                 # poison destination low
sd   t2, 168(a1)                 # poison destination HIGH  <- the value that must be overwritten
sd   t0, 128(a1)                 # source low  = non-zero
sd   x0, 136(a1)                 # source high = ZERO
LDC(a6, a1, 128)        # ldc a6, 128(a1)
STC(a1, a6, 160)        # stc a6, 160(a1)
ld   s6, 160(a1)        # ARM C low  -- expect 0123456789abcdef
ld   s7, 168(a1)        # ARM C high -- 0 = copy worked; deadbeefcafef00d = STALE, hole confirmed

# ---------------------------------------------------------------- ARM D: pointer provenance
# The monitor's stack capability is not available here (bare-metal M-mode, no cscratch symbol),
# so reproduce the TRANSITION the entry glue performs rather than its source: a LINEAR stack-like
# capability, delin'd, then offset exactly as compiled code derives a data pointer. That is the
# sequence whose OUTPUT type is the open question.
MKCAP(s8, CAP_TYPE_LIN)
CAPPRINT(s8)            # ARM D-1 -- LINEAR (1) as fabricated, proves the arm is live
DELIN(s8)                        # the glue's delin
CAPPRINT(s8)            # ARM D-2 -- expect NONLIN (2)
li   t3, 0x40
CINCOFFSET(s9, s8, t3)           # a compiler-style derived data pointer
CAPPRINT(s9)            # ARM D-3 -- the type an ordinary data pointer actually carries

# ---------------------------------------------------------------- done
  li   a0, 1
  lla  a1, tohost
  sd   a0, 0(a1)
1:
  j 1b

.section .data
.align 6
test_buf_start:
  .zero 512
test_buf_end:

.align 6
.globl tohost
tohost: .dword 0
.globl fromhost
fromhost: .dword 0
