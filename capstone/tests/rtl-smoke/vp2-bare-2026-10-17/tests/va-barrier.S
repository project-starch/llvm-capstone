# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-barrier.S  (vp2 item D, step D4: the table barrier, the semantics owner's ruling of 2026-10-17)
#-----------------------------------------------------------------------------
#
# The RTL diverges from QEMU's sequential table: a plain load of a table word younger than a non-ending table writer
# (CSNODE, DELIN, CDERIVE; SPLIT, MREV, INIT likewise) can execute before the rev-node unit's write lands and read the
# PREVIOUS word; the load is ordered after the writer only across a serialising event (a CSR write, an escape, or one
# of the flushing ops REVOKE, DROP, CREVOKE, CSRETIRE). The rule for software: put a serialising CSR write in front of a
# table read that follows such a writer. This arm gives the rule a reading, in va-runv6's setup (M under csupctl.table):
# after CDERIVE the header's next-unused read at once is the old 5, and after a CSR write it is 6: that pair is the
# rule's reading. The outcome without the barrier is a race and is recorded, not promised: after DELIN the child's
# word 1 read FRESH in this arm in both shapes tried (a load one instruction after DELIN, and va-runv6's li/cincoffset/ld),
# while va-runv6 itself read it stale (LINEAR still set) with other instructions around; the arm keeps va-runv6's shape
# and records the fresh word.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   4 (CSNODE's ancestor); rec3 w1 at once 0x1000000100000001 (fresh here; stale in va-runv6's first run); after the
#   barrier 0x1000000100000001; next-unused at once 5 (stale); after the barrier 6.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSNODE(rd, rs1)      .insn r 0x5B, 0x1, 0x54, rd, rs1, x0

#define SREVROOT 0x5C1
#define CSUPCTL  0x7C4

.option norvc

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

# the table: RECYCLE, capacity 64, next 3, record 2 valid
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 64
sd t0, 0(s8)
li t0, 3
sd t0, 8(s8)
sd x0, 16(s8)
sd x0, 24(s8)
li t0, 32
CINCOFFSET(t1, s8, t0)
sd x0, 0(t1)
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)
MINT(s2, CAP_TYPE_LIN, data, data_end, CAP_PERM_RW)
lla t0, tbl
csrw SREVROOT, t0
csrsi CSUPCTL, 8               # M's table namespace

CSNODE(a2, s2)                 # child 3, ancestor 4
csrw mscratch, x0              # barrier
CAPPRINT(a2)                   # 4
DELIN(s2)
li t0, 48
CINCOFFSET(t1, s8, t0)         # record 3 (va-runv6's shape: the load two instructions after the writer)
ld t2, 8(t1)                   # at once
CAPPRINT(t2)                   # 0x1000000100000001 here (fresh); va-runv6's first run read the previous word, 0x3000000100000001
csrw mscratch, x0              # the barrier
ld t2, 8(t1)
CAPPRINT(t2)                   # 0x1000000100000001

li t0, 32
CDERIVE(s3, s2, t0)            # node 5: next-unused 5 -> 6
ld t2, 8(s8)                   # at once
CAPPRINT(t2)                   # 5: the previous word
csrw mscratch, x0              # the barrier
ld t2, 8(s8)
CAPPRINT(t2)                   # 6

csrci CSUPCTL, 8
CAP_PASS(s11)

.align 12
trap_handler:
  csrr t0, mcause
  CAPPRINT(t0)                 # no trap is expected
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
data:
  .zero 4096
data_end:
.align 12
tbl:
  .zero 4096
tbl_end:
  TEST_DATA
RVTEST_DATA_END
