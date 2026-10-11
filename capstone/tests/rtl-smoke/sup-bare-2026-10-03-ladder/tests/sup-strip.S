# See LICENSE for license details.

#*****************************************************************************
# sup-strip.S  (supervised CALL on silicon, verification step 0)
#-----------------------------------------------------------------------------
#
# commit_stage's exception_o is not gated by dom_switch_busy_i. One cycle after a CALL's
# handshake the next, younger, speculative head is visible to the exception block while the
# switcher runs; if the decoder marked it with an interrupt, csr_regfile takes a trap in the
# middle of the switch (mstatus.mie/mepc/mcause/priv change) and the switcher then saves the
# trapped mstatus as the caller's. The frontend redirect is suppressed, so the only symptoms are
# a corrupted saved mstatus and a clobbered mepc/mcause.
#
# This test makes a machine software interrupt pending by a store to the CLINT's msip just
# before the CALL, with MSIE and MIE enabled, so the interrupt is marked on an instruction
# decoded while the CALL is in flight. STRIP_DELAY (nops between the msip store and the CALL)
# tunes when the write reaches the CLINT; the batch sweeps it.
#
# Readings: the callee prints mcause and mepc as its first instructions, the caller prints
# them after RETURN. Pre-registered:
#   hazard present (today):  for some STRIP_DELAY the sim-only detector in commit_stage
#       (`SUP_TRACE`: "SUP exception during switch") fires, and the callee prints
#       mcause = 0x8000000000000003 with mepc inside the caller, i.e. a trap was taken during
#       the switch although the callee's image has MIE = 0.
#   strip in place:  the detector never fires at any delay; the callee prints mcause 0;
#       the interrupt stays pending and is taken by the caller after return (mcause printed
#       by the caller's handler = 0x8000000000000003), or by the callee if its image enables it.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#ifndef STRIP_DELAY
#define STRIP_DELAY 0
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

# s1: LINEAR RW over the CLINT (msip at +0)
CAPCREATE(s1)
li a0, CAP_TYPE_LIN
li a2, 0x02000000
li a3, 0x02001000
li a4, CAP_PERM_RWX
CAPTYPE(s1, a0)
CAPBOUND(s1, a2, a3)
CAPPERM(s1, a4)

# a1: the context region; a2: the callee's code capability
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a2, context_start
lla a3, context_end
li a4, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a2, a3)
CAPPERM(a1, a4)
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, _stub
lla a4, _stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)

STC(a1, a2, 0)                 # slot 0: pc capability
li a0, 0x0000000A00040000      # mstatus image: MIE = 0 -- the callee must NOT take the interrupt
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)                  # mie image: 0
SEAL(a3, a1)

# arm: MSIE and MIE in the caller, then make msip pending right before the CALL
li t0, 8
csrs mie, t0
csrsi mstatus, 8
li t0, 1
sw t0, 0(s1)                   # msip := 1 (reaches the CLINT through the write buffer and AXI)
.rept STRIP_DELAY
  nop
.endr
CALL(a3, a3)
nop
nop
nop
nop

# back in the caller: what does the architectural state say?
csrr t2, mcause
CAPPRINT(t2)                   # #4 caller mcause after RETURN
csrr t2, mepc
CAPPRINT(t2)                   # #5 caller mepc after RETURN
sw x0, 0(s1)                   # clear msip
CAP_PASS(s11)

.align 4
trap_handler:
  # the caller's handler: print the cause, clear msip and return (the interrupt must be able to land HERE, after the switch)
  csrr s4, mcause
  CAPPRINT(s4)                 # #3 or #6: the interrupt taken OUTSIDE the switch
  sw x0, 0(s1)
  li t0, 8
  csrc mie, t0
  csrw mcause, x0              # so a later nonzero mcause can only come from a later trap (mcause is not exchanged)
  mret

.align 4
_stub:
  csrr t2, mcause
  CAPPRINT(t2)                 # #1 callee mcause on entry: 0 expected; 0x8000000000000003 = the trap was taken during the switch
  csrr t2, mepc
  CAPPRINT(t2)                 # #2 callee mepc on entry
  la a1, _stub
  RETURN(ra, a1, x0)
_stub_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
context_start:
  .zero 2048
context_end:
  TEST_DATA
RVTEST_DATA_END
