# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-nogrow.S -- NOGROW at core level: a full table that may not grow answers CDERIVE with the no-fault failure (2026-11-26)
#-----------------------------------------------------------------------------
#
# The unit-level case is sublet-life-diff.py's paged-full-nogrow (the unit against the reference's table code). This
# arm takes it through the core. It uses b5b2-sbdrain.S's paged table, FULL (capacity 256, next 256, record page 1
# unpublished), with the header's growth word (word 6, offset 48) bit 0 set: NOGROW (capstone_sublet_life.sv
# S_O_H3, `nogrow_q <= rsp_w0_q[0]`). S runs one CDERIVE of a 16-byte child of the root {0,3}. The reference's answer on
# a full table that cannot grow is the no-fault FAIL: no exception, and rd untagged.
#   -DNG_CTL: the same with the growth word 0. The table can grow, so the CDERIVE raises 30, as b5b2-sbdrain-none
#   does on silicon.
# Prints: the CDERIVE's cause (s1: 0 if no trap), LCC type of its rd (x21, zeroed first: 1 = a tagged child,
# 7 = untagged), the header's next after the CDERIVE (256 = nothing allocated), and the M trap count.
# PRE-REGISTERED (2026-11-26, written before any run), on b42c8c644's core (capstone-ariane b5b2-hangmon-c + this test):
#   NOGROW:  0x0 0x7 0x100 0x0     (no fault, rd untagged, nothing allocated, no trap)
#   NG_CTL:  0x1e 0x7 0x100 0x1    (30; rd unwritten, so still the zero it was given; nothing allocated; one trap)
# FIRST RUN (336f99a85, 2026-11-26, s2pre-wt .sup/ng, dirty=no): AS PREDICTED, both arms. NOGROW 0x0 0x7 0x100 0x0, the
#   unit's derive commit answering status 0 with record id 0 (`SLU op=3 ... status=0 rid=0`: the no-fault FAIL);
#   NG_CTL 0x1e 0x7 0x100 0x1, the same commit answering status 2 (the 30).
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT 0x5C1

#define CS_MINT_INTO(REG, LO, HI) \
  .pushsection .data; .balign 16; 77: .zero 64; .popsection; \
  lla a0, 77b; lla a3, LO; sd a3, 0(a0); lla a4, HI; sd a4, 8(a0); li a5, 0x10f; sd a5, 16(a0); \
  sd a3, 24(a0); sd x0, 32(a0); sd x0, 40(a0); addi a5, a0, 48; .insn r 0x5B, 0x1, 0x50, a3, a5, a0; LDC(REG, a5, 0)

/* the growth, as the kernel does it: (a) when FILL, then (b) and (c); clobbers a2-a5 */
.macro GROW fill
.if \fill
  lla a2, pg_rp1
  li a3, -1
  slli a3, a3, 32
  li a4, 256
1:
  sd x0, 0(a2)
  sd a3, 8(a2)
  addi a2, a2, 16
  addi a4, a4, -1
  bnez a4, 1b
.endif
  lla a2, pg_d2
  lla a3, pg_rp1
  sd a3, 8(a2)                 # (b) directory 2 [1] := record page 1
  lla a2, tbl
  li a3, 3
  slli a3, a3, 62
  ori a3, a3, 512
  sd a3, 0(a2)                 # (c) CAPACITY := 512
.endm

.option norvc
.section .text.init
.globl _start
_start:
li gp, 1
lla t0, trap_handler
csrw mtvec, t0
csrw mcause, x0
csrw medeleg, x0
csrw mideleg, x0
csrw mie, x0
csrw satp, x0
li t0, -1
csrw mcounteren, t0            # rdcycle in S (SB_COST)
li t0, -1
srli t0, t0, 10
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0

# ---------------------------------------------------------------- the table: b5b2-mtrap.S's
lla s8, tbl
li t0, 3
slli t0, t0, 62
ori t0, t0, 256
sd t0, 0(s8)                   # capacity 256 | RECYCLE | PAGED: one record page
li t0, 256
sd t0, 8(s8)                   # next 256: every slot of page 0 taken, the table full
sd x0, 16(s8)
li t0, 4
sd t0, 24(s8)
li t0, 0x32454741
slli t0, t0, 32
li t1, 0x50444f4e
or t0, t0, t1
sd t0, 32(s8)
sd x0, 40(s8)
#ifdef NG_CTL
sd x0, 48(s8)                  # growth 0: the table may grow
#else
li t0, 1
sd t0, 48(s8)                  # growth word bit 0: NOGROW
#endif
sd x0, 56(s8)
lla t0, pg_d1
sd t0, 64(s8)
lla t1, pg_d1
lla t0, pg_d2
sd t0, 0(t1)
lla t1, pg_d2
lla t0, pg_rp0
sd t0, 0(t1)
li t0, -1
slli t0, t0, 32
lla t1, pg_rp0
li t2, 512
1:
sd t0, 8(t1)
addi t1, t1, 16
addi t2, t2, -1
bnez t2, 1b
lla t1, pg_rp0
li t0, 6
slli t0, t0, 34
sd t0, 48(t1)
sd x0, 56(t1)
li t0, 5
slli t0, t0, 34
ori t0, t0, 8
sd t0, 96(t1)
li t0, 6
sd t0, 104(t1)
li t0, 4
slli t0, t0, 34
ori t0, t0, 8
sd t0, 80(t1)
li t0, 12
li t2, 1
slli t2, t2, 62
or t0, t0, t2
sd t0, 88(t1)
li t0, 8
sd t0, 64(t1)
li t0, 10
sd t0, 72(t1)

CS_MINT_INTO(x26, page_a, page_a_end)
ld a3, 8(a5)
srli a4, a3, 1
li a0, 0x7fffffff
and a4, a4, a0
li a0, 3
bne a4, a0, kit_key_bad
srli a4, a3, 32
bnez a4, kit_key_bad
j kit_key_ok
kit_key_bad:
li a0, 0x6b657962616420
CAPPRINT(a0)
RVTEST_PASS
kit_key_ok:
li x28, 0

li s1, 0
lla t0, tbl
csrw SREVROOT, t0
#ifdef SB_SRET
li t0, 1
slli t0, t0, 30
csrw medeleg, t0               # cause 30 to S
lla t0, s_handler
csrw stvec, t0
li t0, 0x1800
csrc mstatus, t0               # MPP = U
lla t0, u_part
#else
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0               # MPP = S
lla t0, s_part
#endif
csrw mepc, t0
lla x30, m_after
mret

m_after:
#ifdef SB_SRET
lla t1, results
lla t2, s_count
ld t2, 0(t2)
sd t2, 0(t1)
sd a6, 8(t1)
sd a7, 16(t1)
#endif
lla t1, results
ld t2, 0(t1)
CAPPRINT(t2)
ld t2, 8(t1)
CAPPRINT(t2)
ld t2, 16(t1)
CAPPRINT(t2)
CAPPRINT(x28)
csrw SREVROOT, x0
RVTEST_PASS

.align 4
s_part:
  li a1, 16
  li s1, 0
  li x21, 0
  CDERIVE(x21, x26, a1)        # the table is full: NOGROW's no-fault FAIL, or 30 (NG_CTL)
  mv a6, s1
  li a7, 0; LCC(a7, x21, x1)   # the type: 1 = a tagged child, 7 = untagged
  ld a5, 8(s8)                 # the header's next
  lla a4, results
  sd a6, 0(a4)
  sd a7, 8(a4)
  sd a5, 16(a4)
  ecall

.align 4
trap_handler:
  csrr t0, mcause
  li t1, 9
  beq t0, t1, 1f
  li t1, 8
  beq t0, t1, 1f
  mv s1, t0
  addi x28, x28, 1
  csrr t1, mepc
  addi t1, t1, 4
  csrw mepc, t1
  mret
1:
  jr x30

.align 12
img_end:
RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.balign 16
results: .zero 32
s_count: .dword 0
.align 12
page_a:
  .zero 4096
page_a_end:
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
RVTEST_DATA_END
