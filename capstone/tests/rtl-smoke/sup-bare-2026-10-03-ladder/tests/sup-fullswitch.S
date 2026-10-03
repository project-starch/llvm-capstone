# See LICENSE for license details.

#*****************************************************************************
# sup-fullswitch.S  (supervised CALL on silicon, verification step 1)
#-----------------------------------------------------------------------------
#
# The domain switcher's FULL exchange (ids 0..66: pc capability, CSRs, CPMPs, x1..x31 with
# metadata, mcause..satp) has never run: CALL and RETURN send is_full = 0. This test runs a
# CALL/RETURN round trip and reads, in the callee, which registers it received, and in the
# caller, which registers came back.
#
#   build with +define+SUP_FORCE_FULL   -> commit forces is_full = 1 on every switch (sim only)
#   build without it                    -> today's 8-register switch
#
# Pre-registered readings (CAPPRINT values in the retirement trace):
#   FULL:  the callee prints the IMAGE values written into the seal's GPR slots -- x5 = 0x5555,
#          x6 = a TAGGED capability (LCC type = NONLIN), x31 = 0x3131 -- and after RETURN the
#          caller prints its own values back: x5 = 0xA5, x6 its tagged capability, x31 = 0x1F1F,
#          with x13 holding the SEALED capability (CAP_TYPE_SEALED = 5, so LCC selector 1 answers 4).
#   8-REG: the callee prints the CALLER's x5/x6/x31 (0xA5, the caller's cap, 0x1F1F): the
#          registers leak across the switch, which is what the full exchange must stop.
# Mutants (positive controls, applied by the variant patcher):
#   revert the capability read-port mux  -> the caller's x6 comes back UNTAGGED (LCC type 7)
#   revert the id-25 read arm            -> the switch never completes (the harness must time out)
#
# Full-mode slot map (bytes from the seal's start): ids 0-2 at +0/+16/+32 (16 B), ids 3-8 at
# +48..+88 (8 B), CPMP0..15 ids 9-24 at +96 (16 B each), id 25 at +352 (reserved, zero),
# x1..x31 ids 26-56 at +368 + 16*(n-1), mcause..satp ids 57-66 at +864 (8 B each), end +944.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define X5_SLOT   (368 + 16*4)
#define X6_SLOT   (368 + 16*5)
#define X31_SLOT  (368 + 16*30)

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0

# capability mode, covering this whole program
lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

# a1: LINEAR RWX over the context region (2048 B >= the 1024 SEAL requires and the 944 full mode uses)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a2, context_start
lla a3, context_end
li a4, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a2, a3)
CAPPERM(a1, a4)

# a2: LINEAR RWX code capability for the callee
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, _stub
lla a4, _stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)

# a7: a NONLIN capability used as the callee's image for x6 (tagged) and as the caller's x6
CAPCREATE(a7)
li a0, CAP_TYPE_NONLIN
li a3, 0x3000
li a4, 0x4000
li a5, CAP_PERM_RWX
CAPTYPE(a7, a0)
CAPBOUND(a7, a3, a4)
CAPPERM(a7, a5)

# the 8-register image (as call-ctx-save.S writes it)
STC(a1, a2, 0)                 # slot 0: the callee's pc capability
li a0, 0x0000000A00040000
sd a0, 48(a1)                  # mstatus: SXL=2, MPP=0, MIE=0
sd x0, 56(a1)                  # mideleg
sd x0, 64(a1)                  # medeleg
sd x0, 72(a1)                  # mip
sd x0, 80(a1)                  # mie
# the full-mode image: three GPR slots the callee will read back
li t0, 0x5555
sd t0, X5_SLOT(a1)
STC(a1, a7, X6_SLOT)           # x6 image: a TAGGED NONLIN capability
li t0, 0x3131
sd t0, X31_SLOT(a1)

SEAL(a3, a1)                   # a3 := SEALED; a1 consumed
CAPPRINT(a3)                   # #1 the sealed capability (type 4)

# the caller's own register signatures
li x5, 0xA5
MOVC(x6, a7)                   # x6 := the caller's tagged capability
li x31, 0x1F1F
li x7, 0x0707
li x29, 0x1D1D

CALL(a3, a3)                   # x13 holds the seal; RETURN hands it back resealed into x13

# back in the caller
addi s2, x5, 0
CAPPRINT(s2)                   # #5 0xA5 (via s2: the trace shows only retiring writes)
CINCOFFSETIMM(s2, x6, 0)
CAPPRINT(s2)                   # #6 the caller's capability's cursor 0x3000 (via s2)
LCC(s3, x6, x1)                # selector 1 = type - 1: 1 = NONLIN (tagged), 7 = NOT_CAP (tag lost)  (s3: t1 IS x6)
CAPPRINT(s3)                   # #7 1
addi s2, x31, 0
CAPPRINT(s2)                   # #8 0x1F1F
addi s2, x7, 0
CAPPRINT(s2)                   # #9 0x0707
addi s2, x29, 0
CAPPRINT(s2)                   # #10 0x1D1D
LCC(s3, x13, x1)
CAPPRINT(s3)                   # #11 4 = x13 came back SEALED (CINCOFFSET refuses a SEALED cap, flu_unit.anvil:32; LCC reads any type)
# a SECOND round trip proves x13 came back SEALED: it only works on a SEALED capability
li x5, 0xA6
CALL(a3, a3)
addi s2, x5, 0
CAPPRINT(s2)                   # #16 after the second trip: FULL 0xA6, 8-REG 0xBAD5
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # any trap: its cause, then stop
  j fail

fail:
  1: j 1b                      # stay here: the harness's cycle ceiling is the FAIL reading

.align 4
_stub:
  addi s2, x5, 0
  CAPPRINT(s2)                 # #2  FULL: 0x5555   8-REG: 0xA5
  CINCOFFSETIMM(s2, x6, 0)
  CAPPRINT(s2)                 # #3  FULL: the image capability's cursor 0x3000 (tagged; untagged -> cause 24 trap print)
  LCC(s3, x6, x1)
  CAPPRINT(s3)                 # #3b FULL: 1 = NONLIN
  addi s2, x31, 0
  CAPPRINT(s2)                 # #4  FULL: 0x3131   8-REG: 0x1F1F
  # scribble, so that a leak back into the caller is visible  (second trip prints #13..#15)
  li x5, 0xBAD5
  li x7, 0xBAD7
  li x29, 0xBAD9
  li x31, 0xBADF
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
