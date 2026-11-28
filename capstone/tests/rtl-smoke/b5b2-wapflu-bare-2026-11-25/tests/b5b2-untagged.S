# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-untagged.S -- CINCOFFSET, CINCOFFSETIMM and SCC on an untagged x[rs1]: the CHERI rule in every mode (2026-11-25)
#-----------------------------------------------------------------------------
#
# The reference (capstone-qemu 9289114e06 op_helper.c :762-779, helper_cscincoffset / _cscincoffsetimm / _csscc): an
# untagged x[rs1] is data, not an error. CINCOFFSET and CINCOFFSETIMM write the untagged integer x[rs1] + offset, SCC
# writes the untagged integer x[rs2], x[rs1] is left as it is, and every access through the result still faults. A
# tagged x[rs2] (CINCOFFSET's, SCC's) still raises 24 first. No mode is checked. The RTL applied the rule only under
# the CSRUNV-era virtual-context flag (capstone_flu_unit.anvil :34, :68, :109), constant 0 under the Sublet format, so
# silicon raised 24 where the reference adds (B5T-g2's x264 and imagick faults). The semantics owner's ruling of
# 2026-11-25: drop the gate. CDERIVE needs no change: under the Sublet format it decodes to CDERIVE_S (decoder.sv
# :1364), which refuses an inexact child (29) in every mode already (capstone_dyn_unit.anvil :639).
# Setup: b5b2-uprot.S's exactly (its paged table, x26/s10 = {0,3} over data, x24/s8 = data as an INTEGER). Every U
# and S trap goes to M, whose handler records mcause in x9/s1, counts the trap in x28/t3 and skips the instruction;
# an ecall returns to M's continuation at x30/t5. Before each operation its destination holds a sentinel: 0x5a5a for
# the CINCOFFSETIMM and SCC results, x24 itself for the CINCOFFSET result (printed as the difference from x24).
# Phase 1, protected U (srevroot = tbl), and phase 2, S (srevroot = tbl), each prints eight values:
#   a0 cause, a1 value of  cincoffsetimm a1, x0, 0x40           (NULL + 0x40)
#   a2 cause, a3 value - x24 of  cincoffset a3, x24, t2 = 0x10   (an integer pointer + 0x10)
#   a4 cause, a5 value of  scc a5, x24, t2 = 0x1234
#   a6  U: cause of  ld 0(a3)  (the result's first use as a base: protected U refuses an untagged base, 24)
#       S: cause of  scc t1, x24, x26  (a tagged x[rs2]: 24 on every tree)
#   a7  U: cause of  cincoffset t1, x24, x26  (a tagged x[rs2]: 24)
#       S: cause of  ld 0(a3)  (S: an untagged base is plain, rule 1: 0)
# Phase 3, M, srevroot = 0 (the physical pool): two causes, a CDERIVE from the root x18 whose child is not representable
#   (cursor data + 4, size 0x1004: a length of 2^12 or more needs the iE = 1 form, whose granule is 8 bytes; the base
#   is not 8-aligned) and its exact control (data + 8, size 0x1008).
# Then the trap count.
# PRE-REGISTERED (2026-11-25, written before any run), memory delay 12:
#   the fix (b5b2-wedgeap + the FLU ungate):
#     phase 1: 0x0 0x40 | 0x0 0x10 | 0x0 0x1234 | 0x18 | 0x18
#     phase 2: 0x0 0x40 | 0x0 0x10 | 0x0 0x1234 | 0x18 | 0x0
#     phase 3: 0x1d 0x0        trap count 4 (U: the ld and the tagged rs2; S: the tagged rs2; M: the inexact child)
#   the base (47d749153, the gate in):
#     phase 1: 0x18 0x5a5a | 0x18 0x0 | 0x18 0x5a5a | 0x18 | 0x18
#     phase 2: 0x18 0x5a5a | 0x18 0x0 | 0x18 0x5a5a | 0x18 | 0x0
#     phase 3: 0x1d 0x0        trap count 10
#   phase 3 is the same on both trees: it confirms that the Sublet CDERIVE refuses an inexact child in plain M. A 0 in
#   its first print would contradict the dispatch reading above.
# FIRST RUN, phase 3 VOID: the child was (data + 4, 0x804). That is EXACT: below 2^12 bytes the iE = 0 form keeps byte
#   granularity (ariane_pkg.sv decompress_bounds, E = 0, B/T[2:0] stored). So the arm never created the condition and
#   read 0x0 0x0, trap count 3. Phases 1 and 2 read exactly as pre-registered on the fix tree. Rewritten as above
#   before any rerun; PRE-REGISTERED unchanged.
# READ (memory delay 12, +R43_TRACE):
#   the fix (47d749153 + the FLU ungate; s2pre-wt .sup/flu2): 0x0 0x40 0x0 0x10 0x0 0x1234 0x18 0x18 | 0x0 0x40 0x0 0x10
#     0x0 0x1234 0x18 0x0 | 0x1d 0x0 | 0x4: AS PRE-REGISTERED, all nineteen.
#   the base (47d749153; s2fix-wt .sup/basef): 0x18 0x5a5a 0x18 0x0 0x18 0x5a5a 0x18 0x18 | 0x18 0x5a5a 0x18 0x0 0x18
#     0x5a5a 0x18 0x0 | 0x1d 0x0 | 0xa: AS PRE-REGISTERED, all nineteen. Phase 3's 29 on both trees confirms CDERIVE_S's
#     unconditional refusal of an inexact child in plain M (capstone_dyn_unit.anvil :639).
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT    0x5C1

#ifdef CAP_MINT_CSMINT
/* CSMINT through a descriptor {base, end, perms 0x10f, cursor = base, 0, 0} and a granule private to this use (in
   .data), then an M ldc of the granule (M through a plain base keeps the tag). Clobbers a0, a3, a4, a5. */
#define CS_MINT_INTO(REG, LO, HI) \
  .pushsection .data; .balign 16; 77: .zero 64; .popsection; \
  lla a0, 77b; lla a3, LO; sd a3, 0(a0); lla a4, HI; sd a4, 8(a0); li a5, 0x10f; sd a5, 16(a0); \
  sd a3, 24(a0); sd x0, 32(a0); sd x0, 40(a0); addi a5, a0, 48; .insn r 0x5B, 0x1, 0x50, a3, a5, a0; LDC(REG, a5, 0)
/* the Sublet format stores no type and no rights: a loaded capability decodes NONLIN with every right */
#define MINT(REG, TYPE, LO, HI, PERM) CS_MINT_INTO(REG, LO, HI)
#else
#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)
#endif

#define CLR  li x9, 0
#define PRINT8 CAPPRINT(a0); CAPPRINT(a1); CAPPRINT(a2); CAPPRINT(a3); CAPPRINT(a4); CAPPRINT(a5); CAPPRINT(a6); CAPPRINT(a7)
/* the three operations on an untagged x[rs1], each with its sentinel; clobbers t2 */
#define ARITH3 \
  CLR; li a1, 0x5a5a; CINCOFFSETIMM(a1, x0, 0x40); mv a0, x9; \
  CLR; mv a3, x24; li t2, 0x10; CINCOFFSET(a3, x24, t2); mv a2, x9; \
  CLR; li a5, 0x5a5a; li t2, 0x1234; SCC(a5, x24, t2); mv a4, x9

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
csrw medeleg, x0
csrw satp, x0
li x28, 0

# PMP entry 0: NAPOT over everything, RWX
li t0, -1
srli t0, t0, 10
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0

# ---------------------------------------------------------------- the table: sublet-ns.S's, PAGED, SL_F4's slot 4
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 3
slli t0, t0, 62
ori t0, t0, 512
sd t0, 0(s8)                   # capacity 512 | RECYCLE | PAGED
li t0, 6
sd t0, 8(s8)                   # next 6
sd x0, 16(s8)                  # no free list
li t0, 4
sd t0, 24(s8)                  # allocations
li t0, 0x32454741
slli t0, t0, 32
li t1, 0x50444f4e
or t0, t0, t1
sd t0, 32(s8)                  # magic NODPAGE2
sd x0, 40(s8)                  # retired 0
sd x0, 48(s8)                  # growth 0
sd x0, 56(s8)
lla t0, pg_d1
sd t0, 64(s8)                  # root entry 0 -> directory 1
lla t1, pg_d1
lla t0, pg_d2
sd t0, 0(t1)                   # directory 1 [0] -> directory 2
lla t1, pg_d2
lla t0, pg_rp0
sd t0, 0(t1)                   # directory 2 [0] -> record page 0
lla t0, pg_rp1
sd t0, 8(t1)                   # directory 2 [1] -> record page 1
li t0, -1
slli t0, t0, 32
lla t1, pg_rp0
li t2, 512
1:
sd t0, 8(t1)                   # the page-initialisation rule: every record's word 1 retired, both pages
addi t1, t1, 16
addi t2, t2, -1
bnez t2, 1b
lla t1, pg_rp0
li t0, 5
slli t0, t0, 34
sd t0, 32(t1)                  # slot 2: next 5, depth 0
sd x0, 40(t1)                  #         generation 0, prev 0
li t0, 4
slli t0, t0, 34
ori t0, t0, 8
sd t0, 80(t1)                  # slot 5: next 4, depth 1
li t0, 4
sd t0, 88(t1)                  #         prev 2
li t0, 3
slli t0, t0, 34
ori t0, t0, 8
sd t0, 64(t1)                  # slot 4: next 3, depth 1
li t0, 10
li t2, 1
slli t2, t2, 62
or t0, t0, t2
sd t0, 72(t1)                  #         prev 5, generation 2^30 (stale for the generation-0 id x27 carries)
li t0, 8
sd t0, 48(t1)                  # slot 3: next 0, depth 1
li t0, 8
sd t0, 56(t1)                  #         prev 4

# ---------------------------------------------------------------- the capabilities, by cderive in the physical pool
MINT(x18, CAP_TYPE_NONLIN, _start, tbl_end, CAP_PERM_RWX)  # pool node 2: {0,2}
lla a3, data
lla a4, _start
sub a3, a3, a4
CINCOFFSET(t5, x18, a3)
li a1, 4096
CDERIVE(x26, t5, a1)           # pool slot 3: {0,3} over data
CDERIVE(x27, t5, a1)           # pool slot 4: {0,4} over data
li t0, 0x77
sd t0, 0(x26)
sd x0, 8(x26)
lla x24, data                  # the integer address
li t0, 4088
CINCOFFSET(x20, x26, t0)       # x20/s4 = x26's cursor + 4088: ld 8(x20) reads data + 4096, one past the bounds
CINCOFFSET(x21, x27, t0)       # x21/s5 = x27's, the same

# ---------------------------------------------------------------- phase 1: protected U, srevroot = tbl
lla t0, tbl
csrw SREVROOT, t0
li t0, 0x1800
csrc mstatus, t0               # MPP = U
lla t0, u_phase1
csrw mepc, t0
lla x30, m_after1
mret
m_after1:
sub a3, a3, x24
PRINT8

# ---------------------------------------------------------------- phase 2: S, srevroot = tbl
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0               # MPP = S
lla t0, s_phase2
csrw mepc, t0
lla x30, m_after2
mret
m_after2:
sub a3, a3, x24
PRINT8

# ---------------------------------------------------------------- phase 3: M, srevroot = 0 (the physical pool)
csrw SREVROOT, x0
lla a3, data
lla a4, _start
sub a3, a3, a4
addi a3, a3, 4
CINCOFFSET(x22, x18, a3)       # the root's cursor at data + 4 (x26 spans only 4096 bytes: too short for an iE=1 child)
li t2, 0x1004
CLR
CDERIVE(x23, x22, t2)          # inexact: 4100 bytes needs iE = 1, whose granule is 8; the base is not 8-aligned
mv a0, x9
addi a3, a3, 4
CINCOFFSET(x22, x18, a3)       # data + 8
li t2, 0x1008
CLR
CDERIVE(x23, x22, t2)          # the exact control: 8-aligned base, 8-multiple length
mv a1, x9
CAPPRINT(a0); CAPPRINT(a1)
CAPPRINT(x28)
RVTEST_PASS

# ---------------------------------------------------------------- the U and S code
.align 4
u_phase1:
  ARITH3
  CLR; ld t1, 0(a3);              mv a6, x9
  CLR; CINCOFFSET(t1, x24, x26);  mv a7, x9
  ecall
.align 4
s_phase2:
  ARITH3
  CLR; SCC(t1, x24, x26);         mv a6, x9
  CLR; ld t1, 0(a3);              mv a7, x9
  ecall

.align 4
trap_handler:
  csrr x9, mcause
  li t0, 8
  beq x9, t0, 1f
  li t0, 9
  beq x9, t0, 1f
  addi x28, x28, 1
  csrr x19, mtval
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
1:
  jr x30

.align 12
img_end:
RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
data:
  .zero 8192
.align 12
tbl:
  .zero 4096
pg_d1:
  .zero 4096
pg_d2:
  .zero 4096
pg_rp0:
  .zero 4096
pg_rp1:
  .zero 4096
tbl_end:
  TEST_DATA
RVTEST_DATA_END
