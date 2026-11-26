# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-ldcexc2.S -- b5b2-ldcexc's question with a CAPABILITY fault: in protected U, an LDC through an untagged base
# (cause 24, raised by the LSU's capability check) behind a store backlog (2026-11-25, the S-35 hunt)
#-----------------------------------------------------------------------------
#
# b5b2-ldcexc (integer M, a misaligned LDC) completed: its port-1 requests stopped after two short bursts. The
# misaligned exception comes from a register on the MMU path, and the oracle's strongest case is the capability check's
# arms (load_store_unit.sv's cap_exception, recomputed every cycle on lsu_ctrl). This arm uses one: b5b2-untagged.S's
# setup, then in protected U NFILL stores through x26/s10 (a capability over data) to distinct lines, one more store,
# then at once an LDC through x24/s8 (data's address as an INTEGER: 24 in protected U). -DLE_CTL loads through x26
# instead (in bounds, live: no fault). Prints: the LDC's cause, the trap count.
# PRE-REGISTERED (2026-11-25, written before any run), b5b2-wedgeap 47d749153 (+SUP_AP_SWEEP), memory delay 12:
#   LE_CTL: completes; 0x0, trap count 0.
# FIRST RUN VOID (both arms at the cycle limit, no activity after cycle 506): the generator that cut this file from
#   b5b2-untagged.S dropped _start and the whole setup with the ARITH3 macro. Regenerated; PRE-REGISTERED unchanged.
#   the arm, IF THE HYPOTHESIS HOLDS: the cycle limit, with port 1 requesting continuously (RQDBG rd_req 010 held),
#     WEDGE-WBUF, and the write buffer full with lookups requested and never acked. IF NOT: completes, 0x18, 1 trap.
# READ (memory delay 12, +SUP_AP_SWEEP):
#   base 28fcbb646 (and 47d749153): the arm WEDGES, the hypothesis holding -- the cycle limit, WEDGE-WBUF "store refused:
#     full=1, lookups done, no memory write pending", and through the LED mux 192 0xd1 (return pending, cache-update req, its
#     ACK 0), 223 0x08 (full), 209 0x10 (load port READ), 229 0x94 / 195 0x80 (the load unit cycling SEND_TAG_LDC <-> IDLE,
#     port 1 requesting every cycle). The control completes, 0x0 0x0, after 33,579 cycles.
#   fix (R-80: a faulting LDC pops): the arm completes, 0x18 with ONE trap, after 33,548 cycles; the control is unchanged
#     (33,579 cycles).
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
#ifndef NFILL
#define NFILL 16
#endif


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

# ---------------------------------------------------------------- warm the lines in M, then protected U
li t1, 0x1234
.set m, 0
.rept NFILL
  ld t2, (m*64+128)(x24)
  .set m, m+1
.endr
fence
lla t0, tbl
csrw SREVROOT, t0
li t0, 0x1800
csrc mstatus, t0               # MPP = U
lla t0, u_phase1
csrw mepc, t0
lla x30, m_after1
mret
m_after1:
CAPPRINT(a0)
CAPPRINT(x28)
csrw SREVROOT, x0
RVTEST_PASS

.align 4
u_phase1:
  CLR
.set m, 0
.rept NFILL
  sd t1, (m*64+128)(x26)
  .set m, m+1
.endr
  sd t1, 64(x26)
#ifdef LE_CTL
  LDC(t4, x26, 0)
#else
  LDC(t4, x24, 0)
#endif
  mv a0, x9
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
