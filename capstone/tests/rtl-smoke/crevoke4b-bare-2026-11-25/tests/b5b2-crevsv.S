# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-crevsv.S -- B5b-2 4(b) part 2: the saved position in the core, with what an S handler does between restarts
# (2026-11-25)
#-----------------------------------------------------------------------------
#
# b5b2-crevint.S's CI_SHANDLER arm (the no-Sstc shape: M timer -> M sets STIP -> S handler), with the M timer re-armed
# TICK ahead after each tick until NT ticks have fired, so the CREVOKE stops and restarts several times. Between the
# stop and the restart the S handler does one thing (the saved-position invalidation list of the semantics owner's
# ruling, 2026-11-25), chosen by a define:
#   CS_NONE    nothing: every restart RESUMES from the saved t
#   CS_DERIVE  a CDERIVE from the root (outside the run): every restart after it scans afresh
#   CS_KILL    a CREVOKE of a sibling of o (x21/s5, derived before o), on the first tick only: the next restart
#              scans afresh, later ones resume
#   CS_SREV    csrw srevroot, tbl2; csrw srevroot, tbl (a switch_mm away and back): every restart scans afresh
#   CS_STRAW   a CSTRAW of an integer to a scratch granule: every restart scans afresh
# The handler does it, then clears STIP through the SBI-shaped ecall and srets.
# Prints as b5b2-crevint.S (crevoke cause, cycles, three ids, dead-o cause, 0, timer traps, at the site, S traps at
# the site, checksum, M trap count). The path of each restart is read from the unit's simulation trail
# (SLU-RESUME / SLU-RESUME-MISS lines and the CREVOKE answers), not from the program.
# PRE-REGISTERED (2026-11-25, written before any run), capstone-ariane b5b2-crevoke4b with part 2, KCHUNK 64, NCH 200,
# NT 6, TICK 150 (b5b2-crevint's TICK 20 landed six of seven ticks before the CREVOKE re-executed), in-tree:
#   CS_NONE, CS_SREV, CS_STRAW: crevoke 0x0, ids 0x80000007 0x800000cf 0x800000ce, dead-o 0x19, the checksum equal to
#     b5b2-crevint ARM 0's (no result depends on the path: the handler changes no table word), S traps at the site
#     equal to timer traps at the site (each stop reaches S), M trap count 0x1.
#   CS_DERIVE: crevoke 0x0, the first id 0x80000007, dead-o 0x19; the others recorded (the handler's records take slots).
#   CS_KILL: crevoke 0x0, dead-o 0x19, M trap count 0x1; the ids recorded.
#   Paths (trail): CS_NONE every restart after a stop RESUME (resumes = stops); CS_DERIVE, CS_SREV, CS_STRAW 0 resumes;
#     CS_KILL resumes = stops - 1. No SLU-RESUME-MISS in any arm.
#   A void run: no stop (timer traps at the site 0).
# READ (710998447, +R43_TRACE, rtl dirty=no):
#   none   0x0 0x41bc 0x80000007 0x800000cf 0x800000ce 0x19 0x0 0x7 0x7 0x2 0xe2b9b216cdcfb1f9 0x1
#   derive 0x0 0x492b 0x80000007 0x800000cf 0x800000ce 0x19 0x0 0xc 0xa 0x3 0xd443b2166d81b1f9 0x1
#   kill   0x0 0x468e 0x80000008 0x800000d0 0x800000cf 0x19 0x0 0xc 0xa 0x3 0x5323b3b7586fc8ee 0x1
#   srev   0x0 0x47f0 0x80000007 0x800000cf 0x800000ce 0x19 0x0 0xc 0xa 0x3 0xe2b9b216cdcfb1f9 0x1
#   straw  0x0 0x46f3 0x80000007 0x800000cf 0x800000ce 0x19 0x0 0x7 0x7 0x2 0xe2b9b216cdcfb1f9 0x1
#   Paths (trail): every arm stopped twice; resumes none 2, derive 0, kill 1, srev 0, straw 0; no MISS. AS PRE-REGISTERED.
#   Values as pre-registered except two relations of mine, both instrument misses (values unaffected, not widened
#   after the fact):
#   - the checksum is not b5b2-crevint ARM 0's: this file's .data adds scratch16 and tbl2, the table's pages move,
#     and the root and directory entries hold page addresses. Within this file none, srev and straw agree.
#   - S traps at the site (2) are not the timer traps at the site (7 / 10): several M ticks land while STIP is
#     already pending, before S takes it.
#   Latency (trail, sim memory delay 12, KCHUNK 64): an interrupt during a resumed chunk to the PARTIAL answer 4,042
#   cycles (none: SLU-IRQ 108928, PARTIAL 112970); during the first, fresh scan 4,952 (that scan was near its end).
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT 0x5C1
#define MTIME    0x200bff8
#define MTIMECMP 0x2004000
#ifndef NCH
#define NCH 200
#endif
#ifndef TICK
#define TICK 150
#endif
#ifndef NT
#define NT 6
#endif
#define CI_SHANDLER
#define RAWST(rs, base, off) .insn s 0x0b, 5, rs, off(base)

#define CS_MINT_INTO(REG, LO, HI) \
  .pushsection .data; .balign 16; 77: .zero 64; .popsection; \
  lla a0, 77b; lla a3, LO; sd a3, 0(a0); lla a4, HI; sd a4, 8(a0); li a5, 0x10f; sd a5, 16(a0); \
  sd a3, 24(a0); sd x0, 32(a0); sd x0, 40(a0); addi a5, a0, 48; .insn r 0x5B, 0x1, 0x50, a3, a5, a0; LDC(REG, a5, 0)

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
csrw mcounteren, t0
csrw scounteren, t0
li t0, -1
srli t0, t0, 10
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
li t0, -1
li t1, MTIMECMP
sd t0, 0(t1)                   # no timer until the site arms it

# ---------------------------------------------------------------- the table: b5b2-mtrap.S's
lla s8, tbl
li t0, 3
slli t0, t0, 62
ori t0, t0, 512
sd t0, 0(s8)
li t0, 7
sd t0, 8(s8)
sd x0, 16(s8)
li t0, 4
sd t0, 24(s8)
li t0, 0x32454741
slli t0, t0, 32
li t1, 0x50444f4e
or t0, t0, t1
sd t0, 32(s8)
sd x0, 40(s8)
sd x0, 48(s8)
sd x0, 56(s8)
lla t0, pg_d1
sd t0, 64(s8)
lla t1, pg_d1
lla t0, pg_d2
sd t0, 0(t1)
lla t1, pg_d2
lla t0, pg_rp0
sd t0, 0(t1)
lla t0, pg_rp1
sd t0, 8(t1)
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
#if defined(CI_MTIMER) || defined(CI_SHANDLER)
li t0, 0x80
csrw mie, t0                   # MTIE: M interrupts are taken in S whatever mstatus.MIE says
#endif
#ifdef CI_SHANDLER
li t0, 0x20
csrw mideleg, t0               # STIP to S
lla t0, s_handler
csrw stvec, t0
#endif
li t0, 0x1800
csrc mstatus, t0
li t0, 0x800
csrs mstatus, t0               # MPP = S
lla t0, s_part
csrw mepc, t0
lla x30, m_after
mret

m_after:
lla t1, tbl
lla t2, tbl_end
li a0, 0
1:
ld a1, 0(t1)
slli a2, a0, 1
srli a0, a0, 63
or a0, a0, a2
xor a0, a0, a1
addi t1, t1, 8
bltu t1, t2, 1b
lla t1, results
.irp k, 0,1,2,3,4,5,6
ld t2, 8 * \k(t1)
CAPPRINT(t2)
.endr
lla t1, ints_total
ld t2, 0(t1)
CAPPRINT(t2)
lla t1, ints_site
ld t2, 0(t1)
CAPPRINT(t2)
lla t1, s_site
ld t2, 0(t1)
CAPPRINT(t2)
CAPPRINT(a0)
CAPPRINT(x28)
csrw SREVROOT, x0
RVTEST_PASS

# ---------------------------------------------------------------- S
.align 4
s_part:
#ifdef CI_SHANDLER
  li a2, 0x20
  csrs sie, a2
  csrsi sstatus, 2             # SIE
#endif
#ifdef CS_KILL
  li a1, 16
  CDERIVE(x21, x26, a1)        # the sibling, slot 7; o takes slot 8 and every id below moves by one
#endif
  li a1, 2048
  CDERIVE(x25, x26, a1)        # o, slot 7
  li a1, 16
  li a4, NCH
1:
  CDERIVE(x24, x25, a1)        # a child of o: directly after o in the list
  addi a4, a4, -1
  bnez a4, 1b
  li x23, 0
  li s1, 0
#if defined(CI_MTIMER) || defined(CI_SHANDLER)
  li a2, MTIME
  ld a3, 0(a2)
  addi a3, a3, TICK
  li a2, MTIMECMP
  sd a3, 0(a2)
#endif
  rdcycle a0
site:
  CREVOKE(x25, x0)
  rdcycle a1
  mv a5, s1                    # crevoke's cause
  sub a6, a1, a0
  li a2, -1
  li a3, MTIMECMP
  sd a2, 0(a3)                 # no more timer
  li a1, 16
  CDERIVE(x20, x26, a1)
  CDERIVE(x21, x26, a1)
  CDERIVE(x22, x26, a1)
  lla a4, results
  sd a5, 0(a4)
  sd a6, 8(a4)
  li a2, 0; LCC(a2, x20, x8); sd a2, 16(a4)
  li a2, 0; LCC(a2, x21, x8); sd a2, 24(a4)
  li a2, 0; LCC(a2, x22, x8); sd a2, 32(a4)
  li s1, 0
  CDERIVE(x24, x25, a1)        # from the dead o
  sd s1, 40(a4)
  li s1, 0

  sd s1, 48(a4)
  li a7, 0
  ecall

#ifdef CI_SHANDLER
.align 4
s_handler:
  csrr a2, sepc
  lla a3, site
  bne a2, a3, 1f
  lla a3, s_site
  ld a2, 0(a3)
  addi a2, a2, 1
  sd a2, 0(a3)
1:
#if defined(CS_DERIVE)
  li a2, 16
  CDERIVE(x23, x26, a2)        # outside the run: a child of the root
#elif defined(CS_KILL)
  lla a3, s_site
  ld a2, 0(a3)
  li a3, 1
  bne a2, a3, 5f               # the first tick only
  CREVOKE(x21, x0)             # another CREVOKE
5:
#elif defined(CS_SREV)
  lla a2, tbl2
  csrw SREVROOT, a2
  lla a2, tbl
  csrw SREVROOT, a2
#elif defined(CS_STRAW)
  lla a2, scratch16
  li a3, 0x77
  RAWST(a3, a2, 0)             # a CSTRAW of an integer
#endif
  li a7, 0x5b1
  ecall                        # M clears STIP
  li a7, 0
  sret
#endif

# ---------------------------------------------------------------- M
.align 4
trap_handler:
  csrr t0, mcause
  bltz t0, m_irq
  li t1, 9
  bne t0, t1, 2f
  li t1, 0x5b1
  bne a7, t1, 1f
  li t1, 0x20
  csrc mip, t1                 # the SBI-shaped call: clear STIP
  csrr t1, mepc
  addi t1, t1, 4
  csrw mepc, t1
  mret
1:
  jr x30
2:
  mv s1, t0
  addi x28, x28, 1
  csrr t1, mepc
  addi t1, t1, 4
  csrw mepc, t1
  mret
m_irq:
  lla t1, ints_total
  ld t2, 0(t1)
  addi t2, t2, 1
  sd t2, 0(t1)
  csrr t0, mepc
  lla t1, site
  bne t0, t1, 3f
  lla t1, ints_site
  ld t2, 0(t1)
  addi t2, t2, 1
  sd t2, 0(t1)
3:
#ifdef CI_SHANDLER
  li t1, 0x20
  csrs mip, t1                 # forward the tick to S as STIP
  lla t1, ints_total
  ld t2, 0(t1)
  li t1, NT
  li t0, -1
  bgeu t2, t1, 6f
  li t1, MTIME
  ld t0, 0(t1)
  addi t0, t0, TICK
6:
  li t1, MTIMECMP
  sd t0, 0(t1)
#else
  lla t1, ints_total
  ld t2, 0(t1)
  li t1, 4
  li t0, -1
  bgeu t2, t1, 4f
  li t1, MTIME
  ld t0, 0(t1)
  addi t0, t0, TICK
4:
  li t1, MTIMECMP
  sd t0, 0(t1)
#endif
  mret

.align 12
img_end:
RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.balign 16
results: .zero 64
.balign 16
scratch16: .zero 16
.align 12
tbl2: .zero 4096                # a second table base, page-aligned (srevroot keeps bits 63:12)
ints_total: .dword 0
ints_site: .dword 0
s_site: .dword 0
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
