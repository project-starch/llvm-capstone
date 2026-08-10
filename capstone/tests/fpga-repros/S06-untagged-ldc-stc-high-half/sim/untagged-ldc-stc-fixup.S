# See LICENSE for license details.

#*****************************************************************************
# untagged-ldc-stc-fixup.S
#-----------------------------------------------------------------------------
#
# Validates the SOFTWARE WORKAROUND for the defect proved by untagged-ldc-stc-128.S:
# an ldc/stc round trip of PLAIN data keeps the low 8 bytes and zeroes the high 8.
#
# WHY A WORKAROUND IS HARD. memcpy must copy 16-byte chunks with ldc/stc when the chunk
# holds a capability (only that path preserves the tag) and must NOT when it holds plain
# data (that path destroys half of it). Software cannot simply ask which it is: LCC with a
# NOT_CAP operand raises UNEXPECTED_OPERAND before it ever looks at the requested field
# (capstone_dyn_unit.anvil, func LCC -- the cap_type==NOT_CAP branch), so the obvious
# `if (cap_get_tag(v))` test faults on exactly the plain data it is meant to detect, and a
# capability fault in a domain wedges the core rather than trapping.
#
# THE WORKAROUND UNDER TEST. Copy with ldc/stc, then COMPARE the high 8 bytes of source and
# destination and repair only when they differ:
#
#   ldc  v, 0(src)          # preserves a tag if there is one
#   stc  v, 0(dst)
#   ld   x, 8(src)          # high half of the source
#   ld   y, 8(dst)          # high half of what actually landed
#   when x differs from y:  sd x, 8(dst)      <- written as prose: a comment
#                                                starting with # if would be read
#                                                as a CPP directive, since .S files
#                                                go through the preprocessor
#
# It is self-checking, so it needs no tag query:
#   * chunk IS a capability -> the round trip is exact, the halves match, NO sd runs, and the
#     tag survives. This is the arm that could go wrong: an sd into the high half would clear
#     the tag, so the comparison MUST come out equal here.
#   * chunk is PLAIN data   -> the high half came back zero, the halves differ, the sd repairs
#     it, and all 128 bits are correct. There was no tag to lose.
#   * plain data whose high half is genuinely zero -> halves match, no repair needed, correct.
#
# WHAT TO LOOK FOR in the RVFI trace:
#   ARM C (plain data)
#     t3 = 0123456789abcdef   low  half correct
#     t4 = fedcba9876543210   high half REPAIRED   -> workaround works for plain data
#     t4 = 0000000000000000   repair did not happen
#   ARM D (real capability) -- runs LAST because a lost tag faults rather than returns
#     t5 = 1                  the repair was correctly SKIPPED (halves compared equal)
#     t5 = 0                  the repair RAN on a capability, which would have cleared its tag
#     t6 = 0123456789abcdef -- buf[0] read back THROUGH the copied capability. Only a valid
#          capability can serve as a base, so retiring this load proves the tag survived; a
#          lost tag faults here instead and the .iss log shows the exception.
#
# If ARM C passes and ARM D both skips the repair and still addresses memory through the
# copied capability, the workaround is correct on both kinds of chunk and can go into memcpy.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc
.section .text.init
.globl _start
_start:

# Helper-register discipline, as in the sibling tests: a0/a2/a3/a4 are integer helpers only
# and must never have held a capability, or a stale tag turns a plain operand into
# UNEXPECTED_OPERAND. A LINEAR capability MOVES rather than copies, so every use gets its own.
#define MKCAP(CREG)          \
  CAPCREATE(CREG);           \
  li  a0, CAP_TYPE_LIN;      \
  lla a2, test_buf_start;    \
  lla a3, test_buf_end;      \
  li  a4, CAP_PERM_RWX;      \
  CAPTYPE(CREG, a0);         \
  CAPBOUND(CREG, a2, a3);    \
  CAPPERM(CREG, a4);

MKCAP(s0)               # buffer access capability for arm C
MKCAP(s1)               # buffer access capability for arm D
MKCAP(s2)               # the capability used as arm D's PAYLOAD

# ================================================================= ARM C: PLAIN DATA
MOVC(a1, s0)
li   t0, 0x0123456789abcdef
li   t1, 0xfedcba9876543210
sd   t0, 0(a1)                  # 16 bytes of ordinary data at offset 0, tag is 0
sd   t1, 8(a1)

LDC(a7, a1, 0)                  # the copy, exactly as memcpy's middle loop does it
STC(a1, a7, 32)

ld   t2, 8(a1)                  # high half of the SOURCE
ld   t4, 40(a1)                 # high half of the DESTINATION as it actually landed
beq  t2, t4, 1f                 # equal -> nothing to repair
sd   t2, 40(a1)                 # differ -> repair the high half
1:
ld   t3, 32(a1)                 # RESULT low  -- expect 0123456789abcdef
ld   t4, 40(a1)                 # RESULT high -- expect fedcba9876543210 after the repair

# ================================================================= ARM D: REAL CAPABILITY
# The chunk at offset 64 is a genuine capability. The comparison must come out EQUAL so the
# repair is skipped; an sd here would clear the tag. This arm runs last because a lost tag
# makes the LCC below fault instead of returning.
MOVC(a1, s1)
STC(a1, s2, 64)                 # store a real capability at offset 64

LDC(a7, a1, 64)                 # same copy sequence as arm C
STC(a1, a7, 96)

ld   t2, 72(a1)                 # high half of the SOURCE capability's encoding
ld   t5, 104(a1)                # high half of the DESTINATION's encoding
li   t6, 0
beq  t2, t5, 2f                 # equal -> repair skipped, which is what we want
sd   t2, 104(a1)                # if this ever runs it would clear the tag
li   t6, 1
2:
li   t5, 1
xor  t5, t5, t6                 # t5 = 1 when the repair was SKIPPED, 0 when it ran

LDC(s3, a1, 96)                 # load the copied capability back
ld   t6, 0(s3)                  # USE it as a base. Only a valid capability can address
                                # memory, so retiring this with buf[0] proves the tag
                                # survived; a lost tag faults here instead.


# ================================================================= ARM E: PRE-WRITE THEN STC
# The compare-and-repair sequence in arms C/D has a failure mode arms C/D cannot see: for a
# genuine capability the DESTINATION's stored metadata word need not be bit-identical to the
# SOURCE's (the metadata packs reg_id among other fields), and any such difference makes the
# comparison say "differ", runs the repair store, and CLEARS A LIVE TAG. On the board that
# turned a full SQLite run from an error return into a wedge.
#
# This arm tests a sequence with no data-dependent branch at all, which works because of how
# the cache gates the store (wt_dcache_mem.sv: st_wr_cap = |wr_user_i, and when it is 0 only
# the bank matching the store offset is written):
#
#   sd  lo, 0(dst)        plain stores put BOTH halves down correctly and clear the tag
#   sd  hi, 8(dst)
#   ldc v, 0(src)         then the capability-grained copy goes on top:
#   stc v, 0(dst)           - source IS a capability -> metadata non-zero -> BOTH banks
#                             written, tag restored, high half overwritten correctly
#                           - source is PLAIN data   -> metadata reads 0 -> single-bank store,
#                             bank 1 untouched, so our correct high half SURVIVES
#
# Expect:
#   s5 = 0123456789abcdef   plain-data low  half
#   s6 = fedcba9876543210   plain-data HIGH half survived the stc
#   s7 = 0123456789abcdef   buf[0] read THROUGH a capability copied by this sequence, i.e.
#                           the tag survived. A lost tag faults here instead.

MKCAP(s4)
MKCAP(s8)
MKCAP(s9)

# --- E1: plain data through the pre-write sequence
MOVC(a1, s4)
li   t0, 0x0123456789abcdef
li   t1, 0xfedcba9876543210
sd   t0, 128(a1)                # source chunk: plain data
sd   t1, 136(a1)

ld   a5, 128(a1)                # pre-write both halves at the destination
ld   a6, 136(a1)
sd   a5, 160(a1)
sd   a6, 168(a1)
LDC(a7, a1, 128)                # then the capability-grained copy on top
STC(a1, a7, 160)
ld   s5, 160(a1)                # expect 0123456789abcdef
ld   s6, 168(a1)                # expect fedcba9876543210

# --- E2: a REAL capability through the same sequence, tag must survive
MOVC(a1, s8)
STC(a1, s9, 192)                # source chunk: a genuine capability

ld   a5, 192(a1)                # same pre-write, now over a capability
ld   a6, 200(a1)
sd   a5, 224(a1)
sd   a6, 232(a1)
LDC(a7, a1, 192)
STC(a1, a7, 224)

LDC(s10, a1, 224)               # load the copied capability back
ld   s7, 0(s10)                 # USE it as a base -- only a valid capability can address
                                # memory, so retiring this proves the tag survived

# ================================================================= done
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
