# See LICENSE for license details.

#*****************************************************************************
# call-retpc.S  -- where does the caller resume after CALL/RETURN?  (a pre-existing CALL defect)
#-----------------------------------------------------------------------------
#
# ex_stage.sv packs the CALL's parked return pc as `pc_i + 4` when the dyn unit RESPONDS; pc_i is the issue
# stage's pc_o, which only control-flow-class instructions update, and they update it when merely PRESENTED at
# issue. A jump or branch after the CALL therefore overwrites it before the response (R-47). Found 2026-10-01 by
# the supervised-CALL ladder: a compressed jump after a CALL parked CALL + 8 and the caller resumed mid-instruction.
# No corpus CALL is followed by control flow inside the window (nops, li, CAPPRINT), so it was never seen.
#
# Arm 1: CALL, then c.j over two c.li's.   correct: s2 = 0    resume at CALL+8 (past the c.j): s2 = 2
# Arm 2: CALL, then a 4-byte addi.          correct: s2 = 0x12 the addi skipped: s2 = 0x11
# Arm 3: CALL, then a csrr (cannot issue early: the known-good shape).   s2 = 0x21
# Arm 4: a 4-byte jump (0 / 4); arm 5: a compressed addi (0x51 / 0x50); arm 6: a 4-byte taken branch (0 / 6);
# arm 7: a jump two instructions after the call (0 / 7).
# Readings = s2 after each arm (the readings tool also lists arm 3's own csrr), then mcause (0; a trap ends the test).
# Measured 2026-10-01 on the unmodified RTL (8f6a0af98 + a sim-only display): arm 1 = 2, arm 4 = 4 in every layout;
# arm 6 = 6 in the six-arm layout and 0 in this seven-arm one (the extra 4-byte addi shifts fetch alignment, so the
# branch reaches issue before the dyn unit responds in one layout and not in the other); arm 7 = 0 here. The trigger
# is whether a control-flow instruction reaches issue inside the response window, which fetch timing decides.
# With the latch alone on that tree (R-47's fix): every arm correct, parked pcs CALL + 4 throughout.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

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
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)

# arm 1
li s2, 0
CALL(a3, a3)
c.j 1f
c.li s2, 1
c.li s2, 2
1:
addi s3, s2, 0
CAPPRINT(s3)                   # #1: 0 correct; 2 = resumed past the jump (CALL + 8)

# arm 2  (a GENUINE 4-byte addi: without .option norvc the assembler emits c.addi, measured 2026-10-01)
li s2, 0x11
CALL(a3, a3)
.option push
.option norvc
addi s2, s2, 1
.option pop
addi s3, s2, 0
CAPPRINT(s3)                   # #2: 0x12 correct; 0x11 = the addi after CALL was skipped

# arm 3
li s2, 0x21
CALL(a3, a3)
csrr t0, mcause
addi s3, s2, 0
CAPPRINT(s3)                   # #3: 0x21

# arm 4: a 4-byte unconditional jump (is the trigger "control flow" or "compressed"?)
li s2, 0
CALL(a3, a3)
.option push
.option norvc
j 2f
addi s2, s2, 4
.option pop
2:
addi s3, s2, 0
CAPPRINT(s3)                   # #4: 0 correct; 4 = resumed past the 4-byte jump

# arm 5: a compressed ALU op
li s2, 0x50
CALL(a3, a3)
c.addi s2, 1
addi s3, s2, 0
CAPPRINT(s3)                   # #5: 0x51 correct; 0x50 = the c.addi was skipped

# arm 6: a 4-byte conditional branch, always taken
li s2, 0
CALL(a3, a3)
beq x0, x0, 3f
addi s2, s2, 6
3:
addi s3, s2, 0
CAPPRINT(s3)                   # #6: 0 correct; 6 = resumed past the branch

# arm 7: control flow TWO instructions after the call (the window is the dyn unit's response time, not one slot)
li s2, 0
CALL(a3, a3)
.option push
.option norvc
addi s2, s2, 0
addi s2, s2, 0
j 4f
addi s2, s2, 7
.option pop
4:
addi s3, s2, 0
CAPPRINT(s3)                   # #7: 0 correct; 7 = resumed past a jump that was two instructions after the CALL

csrr t0, mcause
CAPPRINT(t0)                   # #8: 0 (the unmodified RTL may show 28 here: a trap taken during a switch, step 0's hazard)
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause is the last reading
  CAP_PASS(s11)

.align 4
stub:
  la a1, stub
  RETURN(ra, a1, x0)
stub_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
seal:
  .zero 2048
seal_end:
  TEST_DATA
RVTEST_DATA_END
