# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# hpm-probe.S -- are the stock CVA6 hardware performance counters usable as they are on bitstream 3 (e7af48457) and
# the ObjBounds variant (603528b8b)? (SPEC lane's request, 2026-11-03)
#-----------------------------------------------------------------------------
#
# Source facts (e7af48457, identical on 603528b8b): PerfCounterEn = 1 (capstone config pkg :74), MHPMCounterNum = 6
# (ariane_pkg :1101) -> mhpmcounter3..8 / mhpmevent3..8; events perf_counters.sv :105-135; an event's "load" / "store"
# is a COMMITTED instruction with fu == LOAD / STORE (:92-93), and LDC / STC decode with fu = CAPSTONE_DYN
# (decoder.sv :1403, :1408) and keep it to commit -- so the prediction is that LDC and STC count as NEITHER.
# Counters 3..8 are programmed here as: 3 = load (5), 4 = store (6), 5 = L1D access cycles (17: data_req on any of
# the three D$ ports, PTW included, counted per CYCLE the request is up), 6 = L1D miss (2: the WT miss unit's MSHR
# allocation, cacheable only), 7 = branch (9, the control: exactly N + 0 on every arm), 8 = issue stall (22).
# Arms, -DHPM=n, N = 256 iterations each, counters zeroed then read as deltas:
#   HPM 1  ld from one doubleword         predict: load N, store 0, miss <= 1, branch N
#   HPM 2  sd to one doubleword           predict: load 0, store N, branch N
#   HPM 3  ldc from one slot              predict: load 0, store 0 (fu CAPSTONE_DYN), L1D access > 0, branch N
#   HPM 4  stc to one slot                predict: load 0, store 0, L1D access > 0, branch N
#   HPM 5  empty loop                     predict: load 0, store 0, L1D access 0, miss 0, branch N
#   HPM 6  HPM 1 with mcountinhibit[8:3] set   predict: every counter 0; mhpmevent3 reads back 5
#   HPM 7  ld striding 16 B (one line each) over 4 KiB at 0x80100000, outside the image   predict: load N, miss N
# Each arm prints the six counter deltas in order 3..8, the mcycle and minstret deltas, mhpmevent3's readback, then
# the trap count and cause (0, 0).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#ifndef HPM
#define HPM 1
#endif
#define N 256

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s4, 0
  li   s10, 0

  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)

  CAPCREATE(s2)
  li   t1, CAP_TYPE_NONLIN
  lla  a2, heap
  lla  a3, heap_end
  CAPTYPE(s2, t1)
  CAPBOUND(s2, a2, a3)
  li   t1, CAP_PERM_ALL
  CAPPERM(s2, t1)

  CINCOFFSETIMM(t3, s2, 0)       # plain-access base: heap + 0
  li   t1, 0x1000
  CINCOFFSET(s0, s2, t1)         # capability slot: heap + 0x1000
  CAPCREATE(s3)                  # fresh region for HPM 7: 0x80100000 .. +0x1000, outside the image, so no prologue,
  li   t1, CAP_TYPE_NONLIN       # loader or test code touches it (a .data region is not cold under the board wrapper,
  li   a2, 0x80100000            # whose prologue loads every data doubleword: 65 of 256 misses in the first kit sim)
  li   a3, 0x80101000
  CAPTYPE(s3, t1)
  CAPBOUND(s3, a2, a3)
  li   t1, CAP_PERM_ALL
  CAPPERM(s3, t1)
  CINCOFFSETIMM(t0, s2, 0x10)    # the capability that STC stores
  STC(s0, t0, 0)                 # warm the slot (tagged) and the plain doubleword
  LDC(a3, s0, 0)
  ld   t4, 0(t3)
  sd   t4, 0(t3)

  li   t1, 5
  csrw 0x323, t1                 # mhpmevent3 = load
  li   t1, 6
  csrw 0x324, t1                 # mhpmevent4 = store
  li   t1, 17
  csrw 0x325, t1                 # mhpmevent5 = L1D access
  li   t1, 2
  csrw 0x326, t1                 # mhpmevent6 = L1D miss
  li   t1, 9
  csrw 0x327, t1                 # mhpmevent7 = branch
  li   t1, 22
  csrw 0x328, t1                 # mhpmevent8 = issue stall
#if HPM == 6
  li   t1, 0x1f8
  csrs 0x320, t1                 # mcountinhibit[8:3]
#endif
  csrw 0xb03, x0
  csrw 0xb04, x0
  csrw 0xb05, x0
  csrw 0xb06, x0
  csrw 0xb07, x0
  csrw 0xb08, x0
  fence
  csrr s5, mcycle
  csrr s6, minstret
  li   t2, N

3:
#if HPM == 1 || HPM == 6
  ld   t4, 0(t3)
#elif HPM == 2
  sd   t2, 0(t3)
#elif HPM == 3
  LDC(a3, s0, 0)
#elif HPM == 4
  STC(s0, t0, 0)
#elif HPM == 7
  ld   t4, 0(s3)
  CINCOFFSETIMM(s3, s3, 16)
#endif
  addi t2, t2, -1
  bnez t2, 3b

  csrr s7, mcycle
  csrr s8, minstret
  csrr a4, 0xb03
  csrr a5, 0xb04
  csrr a6, 0xb05
  csrr a7, 0xb06
  csrr s9, 0xb07
  csrr s11, 0xb08
  sub  s7, s7, s5
  sub  s8, s8, s6
  CAPPRINT(a4)                   # mhpmcounter3: load
  CAPPRINT(a5)                   # mhpmcounter4: store
  CAPPRINT(a6)                   # mhpmcounter5: L1D access cycles
  CAPPRINT(a7)                   # mhpmcounter6: L1D miss
  CAPPRINT(s9)                   # mhpmcounter7: branch (control: N)
  CAPPRINT(s11)                  # mhpmcounter8: issue stall
  CAPPRINT(s7)                   # mcycle delta
  CAPPRINT(s8)                   # minstret delta
  csrr t1, 0x323
  CAPPRINT(t1)                   # mhpmevent3 readback: 5
  CAPPRINT(s10)                  # trap count: 0
  CAPPRINT(s4)                   # last cause: 0
  CAP_PASS(t5)

.align 4
trap_handler:
  csrr s4, mcause
  addi s10, s10, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 14
heap:
  .zero 0x8000
heap_end:
RVTEST_DATA_END
