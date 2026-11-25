# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).
#*****************************************************************************
# b5b2-crevint.S -- B5b-2 4(b): CREVOKE interrupted in the core (the R-43 replay of the same pc, the trap's epc, the
# release order, a handler's CDERIVE between executions) (2026-11-25)
#-----------------------------------------------------------------------------
#
# llvm-capstone capstone/docs/plans/2026-11-24-interruptible-crevoke.md, the RTL tests: a CREVOKE of a run longer than
# the chunk, with and without interrupts. The unit's semantics are checked against the reference (capstone-qemu
# da49602d76) by sublet-life/crevoke-chunk-diff.py; this test checks the core's path: the PARTIAL becomes a replay, the
# interrupt is taken with epc = the CREVOKE, the CREVOKE re-executes and completes, and the final table is the one an
# uninterrupted CREVOKE leaves.
# Setup (M): b5b2-srvcost.S's table (b5b2-sref.S's paged table in its CSMINT keying, srevroot = tbl) and CSMINT's root
# {0,3} over page_a in x26/s10; bare translation, PMP NAPOT over everything. S: o = cderive x25/s9 <- x26, 2048 (slot 7);
# NCH children of o (a fan, slots 8 .. 7 + NCH, each over 16 bytes), so the list is o, 7+NCH, ..., 8; then
#   rdcycle; site: crevoke x25; rdcycle
# then three cderives from the root (x20/s4, x21/s5, x22/s6; their ids by lcc 8: the free list's first three slots) and
# a cderive from the dead o (its cause). M computes a checksum of the table's five pages (rotate left 1, xor each
# doubleword) and prints.
# Register use, both spellings: the M handler's temporaries t0/x5, t1/x6, t2/x7, its cause record s1/x9 and counter
# t3/x28, its continuation t5/x30; the S handler's temporaries a2/x12, a3/x13; S around the site keeps nothing in
# t0-t2. The M handler takes every trap (medeleg 0): an exception records mcause in s1/x9, counts it in t3/x28 and
# skips the instruction; an S ecall with a7 = 0x5b1 clears mip.STIP and returns (an SBI call's shape); any other
# ecall returns to M's continuation in t5/x30.
#   ARM 0 (default): no interrupt.
#   -DCI_MTIMER: the M timer is armed TICK ahead just before the site and re-armed by the M handler TICK ahead until
#     it has fired four times; the handler counts every timer trap and those whose mepc is the site.
#   -DCI_LAT: CI_MTIMER with TICK 150 and six ticks, for silicon: each timer trap records mtime - mtimecmp at the
#     handler's entry (the stop, the replay and the trap, in mtime ticks), and M first times a 10,000-iteration loop in
#     mcycle and in mtime ticks (the calibration). A trap at the site whose tick fell in the first execution's
#     forward scan measures the scan (O(n), the bound delivered); one during a resumed restart measures the chunk.
#     Prints after the others: lat[0..7] (0 where no trap), then the calibration's cycles and ticks.
#   -DCI_SHANDLER: the no-Sstc shape. One M timer trap; M sets mip.STIP (mideleg STIP, sie STIE, sstatus.SIE) and
#     returns; S takes STIP, its handler counts sepc == the site, derives x23/s7 from o (a record inserted inside the
#     run between executions), clears STIP by the ecall above and srets.
# Prints: crevoke's cause, its cycles, the three ids, the dead-o cause, the dead-x23 cause (CI_SHANDLER, else 0),
#   timer traps, timer traps at the site, S traps at the site, the checksum, the M trap count.
# PRE-REGISTERED (2026-11-25, written before any run), capstone-ariane b5b2-crevoke4b (4(b) part 1 on 904cd9c6b,
# KCHUNK 64), in-tree, NCH 200 (the table's 512 records stay above S's 256-record growth reserve):
#   ARM 0:  0x0 <cycles> 0x80000007 0x800000cf 0x800000ce 0x19 0x0 0x0 0x0 0x0 <checksum> 0x1
#           (release from the end: 8, 9, ..., 207, then o; the free list pops o, 207, 206, each at generation 1; the
#           old list-order release would give 0x80000007 0x80000008 0x80000009. The M trap count is the dead-o cderive.)
#   CI_MTIMER: the same values except cycles (larger), timer traps <recorded, at most 4>, timer traps at the site >= 1,
#           and the SAME checksum as ARM 0 (no result depends on where an execution stopped).
#   CI_SHANDLER: crevoke 0x0; the first id 0x80000007 (o, released last); dead-o 0x19; dead-x23 0x19 (the handler's
#           record, inside the run, is released by the restarted CREVOKE); timer traps 0x1, at the site 0x1; S traps
#           at the site 0x1; M trap count 0x2 (the two cderives from dead records). The other ids and the checksum
#           are recorded (they depend on the chunk the interrupt stopped after).
#   A void run: CI_MTIMER or CI_SHANDLER with no trap at the site (the timer never landed inside the CREVOKE).
#   CI_LAT (pre-registered 2026-11-25 before any run, on 88db40027): the values as CI_MTIMER's (ids, checksum); the
#     latencies and the calibration RECORDED in sim; on silicon, each latency of a trap at the site after the first
#     converted to cycles at most 15,000 (sim's 4,042 at memory delay 12, with room for DRAM's latency).
# READ (e3ba8aed0 = 4(b) part 1, +R43_TRACE, rtl dirty=no), against 904cd9c6b's three core files on the same tests:
#   ARM 0:       0x0 0x37f0 0x80000007 0x800000cf 0x800000ce 0x19 0x0 0x0 0x0 0x0 0xe2b9b216cdcfa989 0x1  (as pre-registered)
#                base 0x0 0x3534 0x80000007 0x80000008 0x80000009 ... 0x3365444f5c1d4a9: the list-order release, the control
#   CI_MTIMER:   0x0 0x412b, the same ids, 0x19 0x0, timer traps 0x7, at the site 0x7, 0x0, the SAME checksum, 0x1.
#                MISSED its bound: 7 timer traps, not at most 4. The trail has ONE PARTIAL answer; every trap's mepc is the
#                site, so six were taken at the re-fetched CREVOKE before it executed again (TICK 20 is shorter than an M
#                round trip, and MTIP stays high until the handler's mtimecmp store lands). Values and checksum as
#                pre-registered. base: timer traps 4, none at the site (the base CREVOKE is not interruptible).
#   CI_SHANDLER: 0x0 0x425f 0x80000007 0x100000047 0x800000cf 0x19 0x19 0x2 0x2 0x1 0x9227b3d4c7c6abc9 0x2: as
#                pre-registered except timer traps 0x2 and at the site 0x2 (predicted 1 and 1; the same MTIP latency).
#                One PARTIAL in the trail. 0x100000047 is slot 71 at generation 2: the handler's record took the head of
#                the first chunk's releases (8..71) and was released again by the restart.
#                base: crevoke's print reads 0x19, a test artefact (the base CREVOKE completes, the interrupt lands at the
#                next instruction, and the handler's cderive from the dead o writes s1 before it is read).
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SREVROOT 0x5C1
#define MTIME    0x200bff8
#define MTIMECMP 0x2004000
#ifndef NCH
#define NCH 200
#endif
#ifdef CI_LAT
#define CI_MTIMER
#ifndef TICK
#define TICK 150
#endif
#define NTICK 6
#else
#define NTICK 4
#endif
#ifndef TICK
#define TICK 20
#endif

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

#ifdef CI_LAT
# the calibration: cycles and mtime ticks over one 10,000-iteration loop, to turn a latency in ticks into cycles
csrr a0, mcycle
li a2, MTIME
ld a1, 0(a2)
li a3, 10000
5:
addi a3, a3, -1
bnez a3, 5b
csrr a4, mcycle
li a2, MTIME
ld a5, 0(a2)
sub a4, a4, a0
sub a5, a5, a1
lla a2, calib
sd a4, 0(a2)
sd a5, 8(a2)
#endif
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
#ifdef CI_LAT
lla t1, lat
.irp k, 0,1,2,3,4,5,6,7
ld t2, 8 * \k(t1)
CAPPRINT(t2)
.endr
lla t1, calib
ld t2, 0(t1)
CAPPRINT(t2)
ld t2, 8(t1)
CAPPRINT(t2)
#endif
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
#ifdef CI_SHANDLER
  CDERIVE(x24, x23, a1)        # from the handler's record
#endif
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
  li a2, 16
  CDERIVE(x23, x25, a2)        # a record inside the run, between executions
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
#ifdef CI_LAT
  li t1, MTIME
  ld t0, 0(t1)
  li t1, MTIMECMP
  ld t2, 0(t1)
  sub t0, t0, t2               # mtime ticks from the deadline to this handler (the stop + the replay + the trap)
  lla t1, ints_total
  ld t2, 0(t1)
  sltiu t1, t2, 8
  beqz t1, 6f                  # lat has 8 entries (more traps than ticks were seen: crevint-mt read 7)
  slli t2, t2, 3
  lla t1, lat
  add t2, t2, t1
  sd t0, 0(t2)                 # lat[trap number]
6:
#endif
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
  csrs mip, t1                 # forward the tick to S as STIP, one shot
  li t0, -1
  li t1, MTIMECMP
  sd t0, 0(t1)
#else
  lla t1, ints_total
  ld t2, 0(t1)
  li t1, NTICK
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
calib: .zero 16
lat: .zero 64
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
