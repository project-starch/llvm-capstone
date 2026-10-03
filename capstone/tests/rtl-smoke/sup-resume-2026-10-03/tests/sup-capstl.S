# sup-capstl.S -- does a supervised RESUME survive an escape that lands right after a capability store-to-load?
#
# Board boot supmon-c5u (2026-10-03): under the FPGA monitor, the P1 speedtest resumed 212 times and then the 213th
# resume CALL never completed (no return, no escape, commit pc frozen before the CALL). The escape that preceded it
# landed directly after this sequence on the domain stack (sqlite3VdbeExec, VA 0x61850..0x61864):
#     stc  a3, 0(G)        a capability store to granule G
#     sb   a1, -1(G)       a byte store into the NEIGHBOURING granule
#     ldc  a2, 0(G)        the capability reloaded from G
#     lbu  a2, 0x8c(a2)    a load through it
# This test makes every escape land near that sequence. A supervised domain loops on it ITER times under a small
# quantum (QUANTUM cycles), and the monitor resumes every preemption, like the FPGA monitor's classic-call hook.
# Build -DINTLOOP for the matched control: the same loop with the capability store/load replaced by sd/ld, i.e. no
# tagged write and no tagged reload; -DNOPLOOP replaces all four memory instructions by addi/andi (the RTL lane's
# control: the arms then differ only in the memory traffic).
# CACHE MISSES (added after the hot-cache session completed every arm): a walk load reads the shadow tag only on an
# L1 MISS, and that read waits in TAG_WAIT behind every outstanding tag write. The hot loop never misses.
#   -DEVICT   the domain also loads one 16-byte line of a 64 KiB buffer per iteration, so a large quantum turns L1
#             (32 KiB, 8-way, 16-B lines) over between escapes, as SQLite's working set does;
#   -DMEVICT  the monitor sweeps the 64 KiB buffer before every resume, so every RESTORE read misses.
#   -DMSWAP   the monitor wraps every CALL in the FPGA monitor's generated __domcallsaves sequence: CPMP0..15 read,
#             cleared and stored with STC, 8 more tag-setting STCs, mcause/mtval/stvec/scause/stval/sepc/sscratch/
#             satp/0x803 and cepc swapped out, cscratch := sp, sp := 0 -- and all of it swapped back after the CALL.
#
# Readings (CAPPRINT, in order):
#   1 csnodefree at the start (reported)
#   2 csupstatus after the last CALL: 1 = kind 0, valid (the domain RETURNed)
#   3 the domain's stored iteration count = ITER (exactly-once: a skipped or doubled iteration moves it)
#   4 the domain's stored checksum = sum of (i & 0xff) for i < ITER: 0x7F800 for ITER 4096, 0x7F80000 for 2^20
#   5 escapes, i.e. resumes (reported; > 0)
#   6 csnodefree at the end (reported)
#   7 mcause = 0 (no trap taken)
#   8 0x5E5E (end marker)
# A HANG reports nothing but "SUPTEST BEGIN", "SB1" and one '.' per 256 escapes, so the dots say how far it got.
# Registers: the board recorder uses x8, x16, x17 (build.sh); the progress print uses x25 (UART cap), x30, x31.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc                  /* like the speedtest domain: no compressed code, every resume point 4-byte */

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSNODEFREE  0xFC4

#ifndef QUANTUM
#define QUANTUM 64
#endif
#ifndef ITER
#define ITER 4096
#endif

#ifdef MSWAP
/* s10: the swap area (512 B, NONLIN RW). Layout: CPMP0..15 at 0..255, 8 tag-setting copies of s11 at 256..383,
   the plain CSRs at 384.., cepc at 464. */
.macro SWAP_OUT
  .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    CCSRRW(t1, (0x10 | \n), x0)
    STC(s10, t1, (16 * \n))
  .endr
  .irp n,0,1,2,3,4,5,6,7
    STC(s10, s11, (256 + 16 * \n))
  .endr
  csrrw t1, mcause, x0
  sd t1, 384(s10)
  csrrw t1, mtval, x0
  sd t1, 392(s10)
  csrrw t1, stvec, x0
  sd t1, 400(s10)
  csrrw t1, scause, x0
  sd t1, 408(s10)
  csrrw t1, stval, x0
  sd t1, 416(s10)
  csrrw t1, sepc, x0
  sd t1, 424(s10)
  csrrw t1, sscratch, x0
  sd t1, 432(s10)
  csrrw t1, satp, x0
  sd t1, 440(s10)
  csrrw t1, 0x803, x0
  sd t1, 448(s10)
  CCSRRW(t1, CCSR_CEPC, x0)
  STC(s10, t1, 464)
  li t1, 0
  CCSRRW(x0, CCSR_CSCRATCH, sp)
  li sp, 0
.endm
.macro SWAP_IN
  CCSRRW(sp, CCSR_CSCRATCH, x0)
  LDC(t1, s10, 464)
  CCSRRW(x0, CCSR_CEPC, t1)
  .irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    LDC(t1, s10, (16 * \n))
    CCSRRW(x0, (0x10 | \n), t1)
  .endr
  ld t1, 384(s10)
  csrrw x0, mcause, t1
  ld t1, 392(s10)
  csrrw x0, mtval, t1
  ld t1, 400(s10)
  csrrw x0, stvec, t1
  ld t1, 408(s10)
  csrrw x0, scause, t1
  ld t1, 416(s10)
  csrrw x0, stval, t1
  ld t1, 424(s10)
  csrrw x0, sepc, t1
  ld t1, 432(s10)
  csrrw x0, sscratch, t1
  ld t1, 440(s10)
  csrrw x0, satp, t1
  ld t1, 448(s10)
  csrrw x0, 0x803, t1
.endm
#else
.macro SWAP_OUT
.endm
.macro SWAP_IN
.endm
#endif

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

# x25: the UART, for the progress dots
CAPCREATE(x25)
li a0, CAP_TYPE_LIN
li a3, BOARD_UART
li a4, BOARD_UART + 0x1000
li a5, CAP_PERM_RW
CAPTYPE(x25, a0)
CAPBOUND(x25, a3, a4)
CAPPERM(x25, a5)

# s5: the monitor's private save area
CAPCREATE(s5)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s5, a0)
CAPBOUND(s5, a3, a4)
CAPPERM(s5, a5)

# s1: NONLIN RW over the 64-byte buffer the domain works in (inherited at first entry, carried by the walks)
CAPCREATE(s1)
li a0, CAP_TYPE_NONLIN
lla a3, buf
lla a4, buf_end
li a5, CAP_PERM_RW
CAPTYPE(s1, a0)
CAPBOUND(s1, a3, a4)
CAPPERM(s1, a5)
sd x0, 0(s1)
sd x0, 8(s1)

#ifdef MSWAP
CAPCREATE(s10)
li a0, CAP_TYPE_NONLIN
lla a3, swaparea
lla a4, swaparea_end
li a5, CAP_PERM_RW
CAPTYPE(s10, a0)
CAPBOUND(s10, a3, a4)
CAPPERM(s10, a5)
#endif
#if defined(EVICT) || defined(MEVICT) || defined(MSWAP)
# s11: NONLIN RW over the 64 KiB eviction buffer (the domain inherits it; the monitor sweeps it under MEVICT)
CAPCREATE(s11)
li a0, CAP_TYPE_NONLIN
lla a3, bigbuf
lla a4, bigbuf_end
li a5, CAP_PERM_RW
CAPTYPE(s11, a0)
CAPBOUND(s11, a3, a4)
CAPPERM(s11, a5)
#endif

# a2: the stub's code capability; a1 -> the seal
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub
lla a4, stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal
lla a4, seal_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)                  # mstatus image: MIE = 0
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)

csrr t0, CSNODEFREE
CAPPRINT(t0)                   # 1  csnodefree at the start
li t0, QUANTUM
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, 0
li s7, 0                       # escapes

CSSUPERVISE(t3, a3, s5)
SWAP_OUT
CALL(a3, a3)
SWAP_IN
supervise_loop:
  csrr t0, CSUPSTATUS
  andi t1, t0, 6
  li t2, 2
  bne t1, t2, done             # kind != 1: the return (0) or a fault (2)
  addi s7, s7, 1
  andi t1, s7, 255
  bnez t1, 1f
  BPUTI(46)                    # '.' every 256 escapes
1:
#ifdef MEVICT
  li t4, 0                     # sweep 64 KiB through s11, one load per 16-byte line: L1 holds none of the walks' lines
  li t5, 0x10000
3:
  CINCOFFSET(t6, s11, t4)
  ld t6, 0(t6)
  addi t4, t4, 16
  bne t4, t5, 3b
#endif
  csrwi CSUPCTL, 1             # resume
  CSSUPERVISE(t3, a3, s5)
  SWAP_OUT
  CALL(a3, a3)
  SWAP_IN
  j supervise_loop
done:
CAPPRINT(t0)                   # 2  csupstatus: 1
ld t0, 0(s1)
CAPPRINT(t0)                   # 3  ITER
ld t0, 8(s1)
CAPPRINT(t0)                   # 4  the checksum
addi t0, s7, 0
CAPPRINT(t0)                   # 5  escapes
csrr t0, CSNODEFREE
CAPPRINT(t0)                   # 6  csnodefree at the end
csrr t0, mcause
CAPPRINT(t0)                   # 7  0
li t0, 0x5E5E
CAPPRINT(t0)                   # 8  end marker
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause, then its pc
  csrr s4, mepc
  CAPPRINT(s4)
  CAP_PASS(s11)

.align 4
stub:
  MOVC(s3, x1)                 # park the SEALEDRET
  li x24, ITER                 # the limit   (x22 / x24 / x26: none is s1..s3 or a recorder register)
  li x22, 0                    # iterations
  li x26, 0                    # checksum
  CINCOFFSETIMM(s2, s1, 16)    # G = buf + 16 (granule 1); buf + 15 is in granule 0, its neighbour
#ifdef EVICT
  li t6, 0xFFF0                # the sweep offset mask (64 KiB, 16-byte steps)
  li t3, 0
#endif
2:
#if defined(NOPLOOP)
  addi x0, x0, 0
  addi x0, x0, 0
  addi x0, x0, 0
  andi a4, x22, 255            # the value the lbu would read
#elif defined(INTLOOP)
  sd x24, 0(s2)                # integer store to G
  sb x22, -1(s2)               # byte into the neighbouring granule
  ld a5, 0(s2)                 # integer reload of G
  lbu a4, -1(s2)               # the byte back
#else
  STC(s2, s1, 0)               # capability store to G (s1 is NONLIN: a copy)
  sb x22, -1(s2)               # byte into the neighbouring granule
  LDC(a2, s2, 0)               # the capability reloaded from G
  lbu a4, 15(a2)               # through it: buf + 15, the byte just stored
#endif
#ifdef EVICT
  addi t3, t3, 16
  and t3, t3, t6
  CINCOFFSET(a5, s11, t3)
  ld t5, 0(a5)
#endif
  add x26, x26, a4
  addi x22, x22, 1
  bne x22, x24, 2b
  sd x22, 0(s1)
  sd x26, 8(s1)
  MOVC(x1, s3)
  la a1, stub
  RETURN(ra, a1, x0)
stub_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
seal:
  .zero 2048
seal_end:
.align 6
buf:
  .zero 64
buf_end:
#ifdef MSWAP
.align 6
swaparea:
  .zero 512
swaparea_end:
#endif
#if defined(EVICT) || defined(MEVICT) || defined(MSWAP)
.align 12
bigbuf:
  .zero 65536
bigbuf_end:
#endif
  TEST_DATA
RVTEST_DATA_END
