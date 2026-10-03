# See LICENSE for license details.

#*****************************************************************************
# sup-sealsize.S  (supervised CALL on silicon: S-11's consequence for the switch, in simulation)
#-----------------------------------------------------------------------------
#
# S-11 (tests/fpga-repros/S11-seal-minsize-alignment-inert, registry R-32): capstone_flu_unit.anvil:215
# means `size < 1024 || misaligned` but the Anvil grammar binds `||` tighter than `<` (lib/parser.mly:87-91
# in the compiler image), so the generated logic is `size < (1024 || misaligned)` = `size < 1`, which holds
# only for an empty region. SEAL therefore accepts a region of any size, and the domain switcher writes the
# 8-register context at base+0..+87 with no bounds check, i.e. OUTSIDE a small seal. (QEMU raises on the
# c128-qemu-merge tip since 2026-09-26, INSUF_CAP_PERMS, minimum 528 bytes; the pinned 674cdab03c has it.)
#
# The test seals a SEAL_SIZE-byte LINEAR region followed by 128 bytes of 0xC0C0... canary words and
# CALLs it. The CALLEE reads the canary: that is where the caller's context has just been written. (A
# reading after RETURN is uninformative, measured 2026-10-01: the RETURN exchange writes the callee's
# medeleg/mip/mie images back, and those ARE the canary words the CALL loaded into them.)
#
# Pre-registered readings (CAPPRINT values in the retirement trace):
#   SEAL_SIZE=64, resident RTL:  #1 0xC0C0C0C0C0C0C0C0 (canary before)  #2 4 (SEAL succeeded: type 5-1)
#            in the callee:      #3 canary[0] at +64 = the caller's medeleg image (0)  <-- written past the bounds
#                                #4 canary[1] at +72 = the caller's mip image          <-- written past the bounds
#                                #5 canary[2] at +80 = the caller's mie image (0)      <-- written past the bounds
#                                #6 canary[3] at +88 = 0xC0C0C0C0C0C0C0C0 (the 8-register exchange ends at +88)
#                                #7 the callee's medeleg CSR = the canary pattern (masked to medeleg's writable bits)
#            after RETURN:       #8..#11 canary[0..3] = 0xC0C0... again (the callee's images written back)
#   SEAL_SIZE=64, fixed check:   #1, then a trap: the handler prints mcause 29 (ILLEGAL_OPERAND_VALUE) and exits.
#   SEAL_SIZE=1023: the same trap (R-32: end is exclusive, the old +1 would have admitted it); SEAL_SIZE=1024: as the
#   2048 control (every canary word intact, #7 = 0). The canary is aligned separately, so an odd size shifts nothing.
#   SEAL_SIZE=2048 (control):    #1, #2 = 4, #3..#6 and #8..#11 all 0xC0C0..., #7 = 0 (the context stays inside).
#   SEAL_SIZE=1024 SEAL_OFF=8:   the region starts 8 bytes into the aligned block, size kept at 1024: the trap, 29 (the
#                                alignment clause alone; nothing is planted first, an STC through a misaligned cursor
#                                traps on its own with 6 -- measured).  SEAL_OFF=16 (aligned, 1024): as the 2048 control.
#   SEAL_SIZE=1024 SEAL_CURSOR=960: a second pc capability and mstatus image at +960/+1008 and the cursor moved there
#                                before SEAL. SEAL with cursor := start (2026-10-01 evening): as the 2048 control.
#                                Without it (mutant no-seal-cursor, or the earlier tree): the exchange happens at +960
#                                and spills past the region -- the 64-byte arm's pattern (#3 0, #4 the mip image, #5 0,
#                                #6 canary, #7 the canary pattern).

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#ifndef SEAL_SIZE
#define SEAL_SIZE 64
#endif
#ifndef SEAL_OFF
#define SEAL_OFF 0
#endif
#ifndef SEAL_CURSOR
#define SEAL_CURSOR 0
#endif

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

# a2: LINEAR RWX code capability for the callee
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, _stub
lla a4, _stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)

# s1: LINEAR RWX over the canary words that follow the seal region
CAPCREATE(s1)
li a0, CAP_TYPE_LIN
lla a3, canary
lla a4, canary_end
li a5, CAP_PERM_RWX
CAPTYPE(s1, a0)
CAPBOUND(s1, a3, a4)
CAPPERM(s1, a5)

# a1: LINEAR RWX over exactly the SEAL_SIZE-byte region (starting SEAL_OFF bytes into the aligned block)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, region_body
lla a4, region_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)

#if (SEAL_OFF & 15) == 0
STC(a1, a2, 0)                 # slot 0: the callee's pc capability
li a0, 0x0000000A00040000
sd a0, 48(a1)                  # slot 3: mstatus image, MIE = 0 (the last slot inside a 64-byte region)
#endif
#if SEAL_CURSOR
# the cursor arm: the same two images at +SEAL_CURSOR, then the cursor moved there before the seal; see the header
CAPCREATE(a6)
li a0, CAP_TYPE_LIN
lla a3, _stub
lla a4, _stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a6, a0)
CAPBOUND(a6, a3, a4)
CAPPERM(a6, a5)
CINCOFFSETIMM(a1, a1, SEAL_CURSOR)
STC(a1, a6, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
#endif

ld t0, 0(s1)
CAPPRINT(t0)                   # #1 canary[0] before: 0xC0C0C0C0C0C0C0C0
SEAL(a3, a1)                   # resident RTL: succeeds for any size; fixed check: trap, cause 29
LCC(s3, a3, x1)
CAPPRINT(s3)                   # #2 4 = SEALED
CALL(a3, a3)
# back in the caller: the RETURN exchange has written the callee's images back over the canary
ld t0, 0(s1)
CAPPRINT(t0)                   # #8 canary[0]
ld t0, 8(s1)
CAPPRINT(t0)                   # #9 canary[1]
ld t0, 16(s1)
CAPPRINT(t0)                   # #10 canary[2]
ld t0, 24(s1)
CAPPRINT(t0)                   # #11 canary[3]
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause is the reading (29 = the fixed SEAL check)
  CAP_PASS(s11)

.align 4
_stub:
  # s1 (x9) is not exchanged by the 8-register switch, so the callee still holds the canary capability
  ld t0, 0(s1)
  CAPPRINT(t0)                 # #3 canary[0] (+64 from the region start): the caller's medeleg image?
  ld t0, 8(s1)
  CAPPRINT(t0)                 # #4 canary[1] (+72): the caller's mip image?
  ld t0, 16(s1)
  CAPPRINT(t0)                 # #5 canary[2] (+80): the caller's mie image?
  ld t0, 24(s1)
  CAPPRINT(t0)                 # #6 canary[3] (+88): intact in every arm
  csrr t0, medeleg
  CAPPRINT(t0)                 # #7 the callee's medeleg: loaded from the canary when the seal is 64 bytes
  la a1, _stub
  RETURN(ra, a1, x0)
_stub_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
region_start:
  .zero SEAL_OFF                 # SEAL_OFF > 0: the sealed region starts that far into the aligned block
region_body:
  .zero SEAL_SIZE
region_end:
.align 4                       # the canary must stay 16-byte aligned for odd SEAL_SIZEs (1023: the ld faulted first, measured)
canary:
  .rept 16
  .dword 0xC0C0C0C0C0C0C0C0
  .endr
canary_end:
  TEST_DATA
RVTEST_DATA_END
