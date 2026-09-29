# See LICENSE for license details.

#*****************************************************************************
# lsc-stc-ldc-slot.S
#-----------------------------------------------------------------------------
#
# The instruction shapes CapstoneLiveSourceCopy emits, run on the RTL before any board image
# carries them. The compiler replaces every movc whose source is read again with
#
#     stc src, off(sp)        one 16-byte stack slot per function
#     ldc dst, off(sp)        issued in the very next slot, same granule
#
# and relies on the pair behaving as movc does for every type, except that an untagged
# (integer) source survives. That rests on four RTL facts, each checked below:
#   - stc leaves a NOT_CAP or NONLIN rs2 alone and nulls a LINEAR one;
#   - the ldc issued right behind it reads what the stc wrote, tag included, even when the
#     granule held a tagged capability a moment earlier (the S-10/S-10b family: a load that
#     misses the store that clears a tag);
#   - a second ldc of a LINEAR value reads a cleared granule;
#   - several ldcs from one stc all read the stored value (the slot-reuse shape).
# Plus the S-07 shape: an ldc whose base is the register the previous ldc just wrote.
#
# Detectors. LCC with selector x0 traps (cause 25, UNEXPECTED_OPERAND) on a NOT_CAP operand,
# so a counted trap means "this register holds no capability". LCC with selector x1 answers
# the type minus one for a capability (LINEAR 0, NONLIN 1) and 7 for a NOT_CAP operand.
# Case 0 proves the trap detector fires before anything relies on it.
#
# Every failure path has its own TESTNUM, so tohost says which check failed:
#   1 self-check        2 integer, cold granule       3 integer over a tagged granule
#   4 NONLIN            5 LINEAR: source, reload, second reload
#   6 back-to-back mixed pairs on one granule          7 one stc, three ldcs
#   8 S-07 chain         9 pressure legs (count of bad legs is in bad_legs)
#   10 unexpected trap

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.section .text.init
.globl _start
_start:

li gp, 1
csrwi mie, 0

la t0, trap_handler
csrw mtvec, t0
la t0, trap_count
sw x0, 0(t0)
la t0, bad_legs
sw x0, 0(t0)

# --- Case 0: the trap detector fires on a plain integer --------------------
li t4, 0x1234
LCC(t5, t4, x0)
la t0, trap_count
lw t1, 0(t0)
li t2, 1
bne t1, t2, fail_1
sw x0, 0(t0)

# --- Setup: s0 is the slot base (LINEAR, over slot_area), as sp is ---------
CAPCREATE(s0)
li a0, CAP_TYPE_LIN
lla a2, slot_area
lla a3, slot_area_end
li a4, CAP_PERM_RWX
CAPTYPE(s0, a0)
CAPBOUND(s0, a2, a3)
CAPPERM(s0, a4)

# s3: a NONLIN capability value, bounds 0x3000..0x4000
CAPCREATE(s3)
li a0, CAP_TYPE_NONLIN
li a2, 0x3000
li a3, 0x4000
li a4, CAP_PERM_RWX
CAPTYPE(s3, a0)
CAPBOUND(s3, a2, a3)
CAPPERM(s3, a4)

# --- Case 2: integer through a cold granule --------------------------------
li a1, 0x5000
STC(s0, a1, 0)
LDC(a2, s0, 0)
li t2, 0x5000
bne a1, t2, fail_2
bne a2, t2, fail_2
LCC(t5, a2, x1)
li t2, 7
bne t5, t2, fail_2
la t0, trap_count
lw t1, 0(t0)
bnez t1, fail_2

# --- Case 3: integer over a granule that holds a tagged capability ---------
STC(s0, s3, 16)
LDC(t6, s0, 16)
li a1, 0x6000
STC(s0, a1, 16)
LDC(a2, s0, 16)
li t2, 0x6000
bne a1, t2, fail_3
bne a2, t2, fail_3
LCC(t5, a2, x1)
li t2, 7
bne t5, t2, fail_3
la t0, trap_count
lw t1, 0(t0)
bnez t1, fail_3

# --- Case 4: NONLIN survives in the source and arrives in the destination --
STC(s0, s3, 32)
LDC(s4, s0, 32)
LCC(t5, s3, x1)
li t2, 1
bne t5, t2, fail_4
LCC(t5, s4, x1)
bne t5, t2, fail_4
LCC(t5, s4, x2)
li t2, 0x3000
bne t5, t2, fail_4
LCC(t5, s4, x4)
li t2, 0x4000
bne t5, t2, fail_4
la t0, trap_count
lw t1, 0(t0)
bnez t1, fail_4

# --- Case 5: LINEAR moves: source nulled, first reload LINEAR, second cleared
CAPCREATE(s6)
li a0, CAP_TYPE_LIN
li a2, 0x5000
li a3, 0x6000
li a4, CAP_PERM_RWX
CAPTYPE(s6, a0)
CAPBOUND(s6, a2, a3)
CAPPERM(s6, a4)
STC(s0, s6, 48)
LDC(s7, s0, 48)
LDC(s8, s0, 48)
LCC(t5, s7, x1)
bnez t5, fail_5
LCC(t5, s7, x2)
li t2, 0x5000
bne t5, t2, fail_5
la t0, trap_count
lw t1, 0(t0)
bnez t1, fail_5
LCC(t5, s6, x0)
LCC(t5, s8, x0)
la t0, trap_count
lw t1, 0(t0)
li t2, 2
bne t1, t2, fail_5
sw x0, 0(t0)

# --- Case 6: back-to-back pairs of mixed types through one granule ---------
li a1, 0x7000
li a3, 0x7100
STC(s0, a1, 64)
LDC(a2, s0, 64)
STC(s0, s3, 64)
LDC(s4, s0, 64)
STC(s0, a3, 64)
LDC(a4, s0, 64)
li t2, 0x7000
bne a2, t2, fail_6
LCC(t5, a2, x1)
li t2, 7
bne t5, t2, fail_6
LCC(t5, s4, x1)
li t2, 1
bne t5, t2, fail_6
li t2, 0x7100
bne a4, t2, fail_6
LCC(t5, a4, x1)
li t2, 7
bne t5, t2, fail_6
la t0, trap_count
lw t1, 0(t0)
bnez t1, fail_6

# --- Case 7: one stc, three ldcs (integer), then two ldcs (NONLIN) ---------
li a1, 0x8000
STC(s0, a1, 80)
LDC(a2, s0, 80)
LDC(a3, s0, 80)
LDC(a4, s0, 80)
li t2, 0x8000
bne a1, t2, fail_7
bne a2, t2, fail_7
bne a3, t2, fail_7
bne a4, t2, fail_7
LCC(t5, a4, x1)
li t2, 7
bne t5, t2, fail_7
STC(s0, s3, 80)
LDC(s4, s0, 80)
LDC(s5, s0, 80)
LCC(t5, s4, x1)
li t2, 1
bne t5, t2, fail_7
LCC(t5, s5, x2)
li t2, 0x3000
bne t5, t2, fail_7
la t0, trap_count
lw t1, 0(t0)
bnez t1, fail_7

# --- Case 8: S-07 shape: the copied register is the next ldc base ---------
# s9: NONLIN over target_area; target_area[0] holds s3.
CAPCREATE(s9)
li a0, CAP_TYPE_NONLIN
lla a2, target_area
lla a3, target_area_end
li a4, CAP_PERM_RWX
CAPTYPE(s9, a0)
CAPBOUND(s9, a2, a3)
CAPPERM(s9, a4)
STC(s9, s3, 0)
STC(s0, s9, 96)
LDC(s10, s0, 96)
LDC(s11, s10, 0)
LCC(t5, s11, x1)
li t2, 1
bne t5, t2, fail_8
LCC(t5, s11, x2)
li t2, 0x3000
bne t5, t2, fail_8
la t0, trap_count
lw t1, 0(t0)
bnez t1, fail_8

# --- Case 9: 16 legs of case 3 with plain stores in flight -----------------
# Each leg re-tags the granule, issues four plain stores to other lines, then the integer
# stc/ldc pair. A bad leg is counted rather than failed, so every leg runs.
li s1, 16
leg_loop:
STC(s0, s3, 112)
la t0, pressure_area
sd s1, 0(t0)
sd s1, 64(t0)
sd s1, 128(t0)
sd s1, 192(t0)
addi a1, s1, 0x100
STC(s0, a1, 112)
LDC(a2, s0, 112)
bne a2, a1, leg_bad
LCC(t5, a2, x1)
li t2, 7
bne t5, t2, leg_bad
j leg_next
leg_bad:
la t0, bad_legs
lw t1, 0(t0)
addi t1, t1, 1
sw t1, 0(t0)
leg_next:
addi s1, s1, -1
bnez s1, leg_loop
la t0, bad_legs
lw t1, 0(t0)
bnez t1, fail_9
la t0, trap_count
lw t1, 0(t0)
bnez t1, fail_9

RVTEST_PASS

fail_1:
li TESTNUM, 1
RVTEST_FAIL
fail_2:
li TESTNUM, 2
RVTEST_FAIL
fail_3:
li TESTNUM, 3
RVTEST_FAIL
fail_4:
li TESTNUM, 4
RVTEST_FAIL
fail_5:
li TESTNUM, 5
RVTEST_FAIL
fail_6:
li TESTNUM, 6
RVTEST_FAIL
fail_7:
li TESTNUM, 7
RVTEST_FAIL
fail_8:
li TESTNUM, 8
RVTEST_FAIL
fail_9:
li TESTNUM, 9
RVTEST_FAIL
unexpected_trap:
li TESTNUM, 10
RVTEST_FAIL

.align 4
trap_handler:
csrr t0, mcause
bltz t0, unexpected_trap
li   t1, 25
bne  t0, t1, unexpected_trap
la   t2, trap_count
lw   t3, 0(t2)
addi t3, t3, 1
sw   t3, 0(t2)
csrr t0, mepc
addi t0, t0, 4
csrw mepc, t0
mret

  .data
RVTEST_DATA_BEGIN

.align 4
trap_count:
  .word 0
.align 4
bad_legs:
  .word 0

.align 6
slot_area:
  .zero 256
slot_area_end:

.align 6
target_area:
  .zero 64
target_area_end:

.align 6
pressure_area:
  .zero 256

  TEST_DATA

RVTEST_DATA_END
