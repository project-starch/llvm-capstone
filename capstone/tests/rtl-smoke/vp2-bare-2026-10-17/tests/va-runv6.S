# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-runv6.S  (vp2 item D, step D4: CSNODE and CSRETIRE over the flat table from M and from trusted S, 2026-10-17)
#-----------------------------------------------------------------------------
#
# A RECYCLE table of capacity 64, next-unused 3, no free list, record 2 valid (CAPCREATE's node), at srevroot. M first
# tries both instructions without csupctl.table: illegal (2, 2). Then under the table bit (read back 8): CSNODE on a
# fresh LINEAR capability allocates the child 3 and its MREV ancestor 4 (QEMU helper_csmint's cap_rev_create then
# cap_rev_mrev), rd = 4, the table's next 5 and allocations 2; a store through the re-noded register lands (0x55);
# DELIN clears the child's LINEAR; CDERIVE puts node 5 under it (depth 2); a store through the derived child lands
# (0x66). CSRETIRE 4 walks the junior run (3 and 5 lose VALID, links kept) and unlinks and invalidates 4 (rd 0); loads
# through the two dead nodes fault 25. Rejections, each rd 1 and nothing written: 4 again, 5 (dead), 0 and 1 (the
# header), 6 (next-unused: unallocated), 65536 and 2^40 (outside the index), a run cycling back to the operand, a run
# looping on itself (the visit bound, next-unused); then the same run made straight retires cleanly (rd 0). The reserve:
# with one free identity CSNODE faults 30; with two it allocates 62 and 63. Then S (srevroot set) retires 63 (rd 0) and
# with srevroot cleared CSRETIRE is illegal (2).
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order), M then S:
#   2; 2; 8; 4; 5; 2; rec3 w0 4; rec3 w1 0x3000000100000001; rec4 w0 0x300000000; rec4 w1 0x3000000000000000; 0x55;
#   rec3 w1 0x1000000100000001; 6; 3; rec5 w0 3; rec5 w1 0x10000000C0000002; rec3 w0 0x500000004; 0x66;
#   0 (retire 4); rec4 w0 0; rec4 w1 0x2000000000000000; rec3 w0 0x500000004; rec3 w1 0x100000001; rec5 w1 0xC0000002;
#   25; 25; 1; 1; 1; 1; 1; 1; 1; 1 (cycle); rec6 w1 0x3000000000000000; 1 (self-loop); rec6 w1 0x3000000000000000;
#   0 (clean); rec7 w1 0x180000001; rec6 w1 0x2000000000000000; 30; 63; 64; 5; rec62 w1 0x3000000FC0000001;
#   rec63 w0 0x3E00000000;  S: 0; rec63 w1 0x2000000000000000; rec62 w1 0x2000000FC0000001; 2.
# Record words: w0 = {next[61:32], prev[29:0]}, w1 = {free[62], linear[61], valid[60], parent[45:30], depth[16:0]}.
# TABLE BARRIER: a load younger than a table-writing DYN op that does not end a node (CSNODE, DELIN, CDERIVE; SPLIT,
# MREV, INIT likewise) can execute before the rev-node unit's write lands and read the PREVIOUS word (the first run of
# this arm read rec3's LINEAR still set after DELIN and next-unused 5 after CDERIVE's bump to 6); commit's R-45 flush
# re-executes younger instructions after REVOKE, DROP, CREVOKE and CSRETIRE only. A serialising CSR write between the
# op and the read orders them here; recorded as a divergence from QEMU's sequential table for the semantics owner.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSNODE(rd, rs1)      .insn r 0x5B, 0x1, 0x54, rd, rs1, x0
#define CSRETIRE(rd, rs1)    .insn r 0x5B, 0x1, 0x53, rd, rs1, x0

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

# read record I's two words through s8 into t2 and print them
#define PREC(I) li t0, 16*(I); CINCOFFSET(t1, s8, t0); ld t2, 0(t1); CAPPRINT(t2); ld t2, 8(t1); CAPPRINT(t2)
#define PREC1(I) li t0, 16*(I); CINCOFFSET(t1, s8, t0); ld t2, 8(t1); CAPPRINT(t2)
#define PREC0(I) li t0, 16*(I); CINCOFFSET(t1, s8, t0); ld t2, 0(t1); CAPPRINT(t2)
#define WREC(I, W0, W1) li t0, 16*(I); CINCOFFSET(t1, s8, t0); li t2, W0; sd t2, 0(t1); li t2, W1; sd t2, 8(t1)

# ---------------------------------------------------------------- the table: RECYCLE, capacity 64, next 3, record 2 valid
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

MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)
sd x0, 0(x26)
sd x0, 8(x26)
MINT(s2, CAP_TYPE_LIN, data, data_mid, CAP_PERM_RW)
MINT(s4, CAP_TYPE_LIN, data_mid, data_end, CAP_PERM_RW)

lla t0, tbl
csrw SREVROOT, t0

# PMP and the CPMP for the S-mode part
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
MINT(a1, CAP_TYPE_NONLIN, _start, tbl_end, CAP_PERM_RWX)
CCSRRW(x0, CCSR_CPMP(0), a1)

# ---------------------------------------------------------------- without csupctl.table: illegal
CSNODE(a2, s2)                 # 2
CSRETIRE(a3, x0)               # 2

csrsi CSUPCTL, 8               # M's table namespace
csrr t2, CSUPCTL
CAPPRINT(t2)                   # 8

# ---------------------------------------------------------------- CSNODE: child 3, ancestor 4
CSNODE(a2, s2)
csrw mscratch, x0              # TABLE BARRIER (see the header): a younger load may read a table word before the unit's write lands
CAPPRINT(a2)                   # 4
ld t2, 8(s8)
CAPPRINT(t2)                   # 5
ld t2, 24(s8)
CAPPRINT(t2)                   # 2
PREC(3)                        # 4; 0x3000000100000001
PREC(4)                        # 0x300000000; 0x3000000000000000
li t0, 0x55
sd t0, 0(s2)                   # through the re-noded register
ld t2, 0(x26)
CAPPRINT(t2)                   # 0x55
DELIN(s2)
csrw mscratch, x0              # TABLE BARRIER
PREC1(3)                       # 0x1000000100000001
li t0, 32
CDERIVE(s3, s2, t0)            # node 5 under 3
csrw mscratch, x0              # TABLE BARRIER
ld t2, 8(s8)
CAPPRINT(t2)                   # 6
ld t2, 24(s8)
CAPPRINT(t2)                   # 3
PREC(5)                        # 3; 0x10000000C0000002
PREC0(3)                       # 0x500000004
li t0, 0x66
sd t0, 0(s3)
ld t2, 0(x26)
CAPPRINT(t2)                   # 0x66

# ---------------------------------------------------------------- CSRETIRE 4: the run (3, 5) and the ancestor
CSRETIRE(a3, a2)
CAPPRINT(a3)                   # 0
PREC(4)                        # 0; 0x2000000000000000
PREC(3)                        # 0x500000004; 0x100000001
PREC1(5)                       # 0xC0000002
ld t2, 0(s3)                   # 25
ld t2, 0(s2)                   # 25

# ---------------------------------------------------------------- rejections: rd 1, nothing written
CSRETIRE(a3, a2)
CAPPRINT(a3)                   # 1: dead
li t0, 5
CSRETIRE(a3, t0)
CAPPRINT(a3)                   # 1: dead
CSRETIRE(a3, x0)
CAPPRINT(a3)                   # 1: the header
li t0, 1
CSRETIRE(a3, t0)
CAPPRINT(a3)                   # 1: the header
li t0, 6
CSRETIRE(a3, t0)
CAPPRINT(a3)                   # 1: next-unused, not allocated
li t0, 65536
CSRETIRE(a3, t0)
CAPPRINT(a3)                   # 1: outside the index
li t0, 1
slli t0, t0, 40
CSRETIRE(a3, t0)
CAPPRINT(a3)                   # 1

# a run that cycles back to the operand: 6 -> 7 -> 6
li t0, 8
sd t0, 8(s8)                   # next-unused 8: records 6 and 7 are allocated
WREC(6, 0x700000000, 0x3000000000000000)
WREC(7, 0x600000006, 0x1000000180000001)
li t0, 6
CSRETIRE(a3, t0)
CAPPRINT(a3)                   # 1: corrupt
PREC1(6)                       # 0x3000000000000000: untouched
# a run looping on itself: 7 -> 7, more records than next-unused
li t0, 16*7
CINCOFFSET(t1, s8, t0)
li t2, 0x700000006
sd t2, 0(t1)
li t0, 6
CSRETIRE(a3, t0)
CAPPRINT(a3)                   # 1: corrupt
PREC1(6)                       # 0x3000000000000000: untouched
# the same run made straight: 6 -> 7 -> end (PREC1 moved t1: address record 7 again)
li t0, 16*7
CINCOFFSET(t1, s8, t0)
li t2, 6
sd t2, 0(t1)
li t0, 6
CSRETIRE(a3, t0)
CAPPRINT(a3)                   # 0
PREC1(7)                       # 0x180000001
PREC1(6)                       # 0x2000000000000000

# ---------------------------------------------------------------- the reserve: two identities
li t0, 63
sd t0, 8(s8)                   # one free identity
CSNODE(a2, s4)                 # 30
li t0, 62
sd t0, 8(s8)                   # two
CSNODE(a2, s4)
csrw mscratch, x0              # TABLE BARRIER
CAPPRINT(a2)                   # 63
ld t2, 8(s8)
CAPPRINT(t2)                   # 64
ld t2, 24(s8)
CAPPRINT(t2)                   # 5
PREC1(62)                      # 0x3000000FC0000001
PREC0(63)                      # 0x3E00000000

# ---------------------------------------------------------------- S: CSRETIRE under srevroot, then without it
li t0, 0x1800
csrc mstatus, t0
li t0, 0x0800
csrs mstatus, t0
lla t0, s_code
csrw mepc, t0
mret

.align 12
s_code:
CSRETIRE(a3, a2)               # 63: 0
li a7, 1
ecall
csrw SREVROOT, x0
CSRETIRE(a3, a2)               # illegal: 2
li a7, 2
ecall
1: j 1b

.align 12
trap_handler:
  csrr t0, mcause
  li t1, 9
  beq t0, t1, 5f
  CAPPRINT(t0)                 # the cause of a refused or faulting instruction
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
5:
  li t1, 1
  bne a7, t1, 7f
  CAPPRINT(a3)                 # 0
  PREC1(63)                    # 0x2000000000000000
  PREC1(62)                    # 0x2000000FC0000001
  j 9f
7:
  csrci CSUPCTL, 8             # leave M's table namespace: S cleared srevroot, so the table base is 0 here
  CAP_PASS(s11)
9:
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
  .zero 64
data_mid:
  .zero 4032
data_end:
.align 12
tbl:
  .zero 4096
tbl_end:
  TEST_DATA
RVTEST_DATA_END
