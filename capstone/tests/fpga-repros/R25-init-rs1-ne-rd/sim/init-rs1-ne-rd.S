/*
 * See LICENSE for license details.
 * *****************************************************************************
 * init-rs1-ne-rd.S  (self-checking form, 2026-09-09; the print-only original is 62b09ca92)
 * -----------------------------------------------------------------------------
 * 
 * R-25: does INIT consume its source when rd differs from rs1?
 * 
 * The spec defines INIT as "MOVC rd, rs1" followed by setting rd's type to linear and its cursor
 * to base + val. MOVC writes cnull to a non-NONLIN source, so after INIT with rs1 != rd the source
 * register must be NOT_CAP. capstone_flu_unit.anvil:147 wrote the NEW linear capability to BOTH
 * rs1 and rd (unguarded, where cincoffset/scc/tighten guard on rs1 == rd): two linear capabilities
 * with one revnode id.
 * 
 * DETECTOR: CINCOFFSET(reg, reg, zero) raises UNEXPECTED_OPERAND (mcause 25) iff reg is NOT_CAP,
 * UNEXPECTED_CAP_TYPE (27) on an UNINIT/SEALED reg, and is a no-op on a LIN/NONLIN reg
 * (capstone_flu_unit.anvil:30-34, :41-42). A trap handler counts traps and records the last cause
 * per phase; the verify block turns the table into one exit code.
 * 
 * PHASES (counts[phase], cause[phase]):
 * 0  detector control: MOVC(s2, a7) on a LIN a7, then probe a7      -> 1 trap, cause 25
 * 1  probe s2 (the MOVC destination)                                -> 0 traps
 * 2  INIT(a6, a5, a1) with a5 UNINIT (valid range, cursor past end), probe a5
 * conformant RTL: 1 trap, cause 25 (source consumed)
 * R-25 RTL:       0 traps          -> exit code 11
 * 3  probe a6                                                       -> 0 traps
 * 4  "is the duplicate usable": STC(a5, s2, 0); LDC(s5, a6, 0); probe s5
 * R-25 RTL:       0 traps (store through the duplicate, load back through rd, a capability)
 * conformant RTL: 2 traps (STC through null a5 -> 25; the slot was never written, LDC yields
 * NOT_CAP, probe -> 25). Recorded in counts, printed, not part of the verdict.
 * 5  INIT(s3, s3, a1) (rd == rs1, the shape every compiled INIT uses); probe s3    -> 0 traps
 * 6  MOVC(s4, s3); probe s3                                          -> 1 trap, cause 25 (it was LIN)
 * 7  probe s4                                                        -> 0 traps
 * 
 * EXIT CODES (tohost exit code = gp): PASS when phases 0,1,2,3,5,6,7 match the conformant column.
 * 10 = phase 0 did not trap (detector dead: nothing below means anything)
 * 11 = phase 2 did not trap: R-25 present
 * 12 = phase 1/3/7 trapped (a destination is not a capability)
 * 13 = phase 5 trapped or phase 6 did not trap (the same-register arm is broken)
 * 20 = a trap with an unexpected cause (anything but 25 in phases 0/2/6, or anything in phase 4
 * other than 25)
 * 
 * The UNINIT operands: MKCAP as LIN over [region+512, region+768) / [region+1024, region+1280),
 * CINCOFFSETIMM by 272 (allowed on LIN; cursor = base+272 > end), then the in-place CAPTYPE
 * (.insn with rs2 = the register itself, so the bounds and cursor survive) to UNINIT. INIT then
 * accepts it (cursor > end) and yields a LIN capability over a real 256-byte window, so phase 4 can
 * store and load through it.
 * 
 */
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"
#define MKCAP(CREG, CTYPE, BASE_OFF, END_OFF) \
  CAPCREATE(CREG);           \
  li  a0, CTYPE;             \
  lla a2, region_start;      \
  addi a2, a2, BASE_OFF;     \
  lla a3, region_start;      \
  addi a3, a3, END_OFF;      \
  li  a4, CAP_PERM_RWX;      \
  CAPTYPE(CREG, a0);         \
  CAPBOUND(CREG, a2, a3);    \
  CAPPERM(CREG, a4);
#define MKUNINIT_PAST_END(CREG, BASE_OFF, END_OFF) \
  MKCAP(CREG, CAP_TYPE_LIN, BASE_OFF, END_OFF)   \
  CINCOFFSETIMM(CREG, CREG, 272);                \
  li  a0, CAP_TYPE_UNINIT;                       \
  .insn r 0x7B, 0x0, 0x5, CREG, a0, CREG;
#define PROBE(REG)  CINCOFFSET(REG, REG, zero)
#define PHASE(N)    li s10, N
.section .text.init
.globl _start
_start:
  la   t0, m_trap_handler
  csrw mtvec, t0
  la   s11, counts
  li   t0, 0
1:sw   zero, 0(s11)
  addi s11, s11, 4
  addi t0, t0, 1
  li   t1, 16
  bne  t0, t1, 1b
  la   s11, counts
# ---------------------------------------------------------------- phase 0/1: detector control
  MKCAP(a7, CAP_TYPE_LIN, 0, 256)
  MOVC(s2, a7)
  PHASE(0)
  PROBE(a7)               # expect trap 25: a7 consumed by MOVC
  PHASE(1)
  PROBE(s2)               # expect no trap
  CAPPRINT(a7)
  CAPPRINT(s2)
# ---------------------------------------------------------------- phase 2/3: THE PROBE (rd != rs1)
  MKUNINIT_PAST_END(a5, 512, 768)
  li   a1, 0
  INIT(a6, a5, a1)
  PHASE(2)
  PROBE(a5)               # conformant: trap 25.  R-25: no trap (a5 is a second LIN capability)
  PHASE(3)
  PROBE(a6)               # expect no trap
  CAPPRINT(a5)
  CAPPRINT(a6)
# ---------------------------------------------------------------- phase 4: is the duplicate usable?
  PHASE(4)
  STC(a5, s2, 0)          # R-25: stores s2 through the duplicate; conformant: trap 25 (a5 is null)
  LDC(s5, a6, 0)          # load it back through rd
  PROBE(s5)               # R-25: no trap (a capability came back); conformant: trap 25
  CAPPRINT(s5)
# ---------------------------------------------------------------- phase 5/6/7: same-register arm
  MKUNINIT_PAST_END(s3, 1024, 1280)
  li   a1, 0
  INIT(s3, s3, a1)
  PHASE(5)
  PROBE(s3)               # expect no trap: s3 is the LIN result
  MOVC(s4, s3)
  PHASE(6)
  PROBE(s3)               # expect trap 25: a LIN source is consumed by MOVC
  PHASE(7)
  PROBE(s4)               # expect no trap
  CAPPRINT(s3)
  CAPPRINT(s4)
# ---------------------------------------------------------------- print the table, then verify
  li   t0, 0
2:slli t1, t0, 2
  add  t1, t1, s11
  lw   t2, 0(t1)          # counts[phase]
  lw   t3, 32(t1)         # cause[phase]
  CAPPRINT(t2)
  CAPPRINT(t3)
  addi t0, t0, 1
  li   t1, 8
  bne  t0, t1, 2b
  lw   t1, 0(s11)         # phase 0
  li   gp, 10
  beqz t1, fail
  lw   t1, 4(s11)         # phase 1
  li   gp, 12
  bnez t1, fail
  lw   t1, 12(s11)        # phase 3
  bnez t1, fail
  lw   t1, 28(s11)        # phase 7
  bnez t1, fail
  lw   t1, 20(s11)        # phase 5
  li   gp, 13
  bnez t1, fail
  lw   t1, 24(s11)        # phase 6
  beqz t1, fail
  lw   t1, 8(s11)         # phase 2 -- the verdict
  li   gp, 11
  beqz t1, fail
  RVTEST_PASS
fail:
  RVTEST_FAIL

.align 4
m_trap_handler:
  csrr t4, mcause
  slli t5, s10, 2
  add  t5, t5, s11
  lw   t6, 0(t5)
  addi t6, t6, 1
  sw   t6, 0(t5)
  sw   t4, 32(t5)
  li   t6, 25
  beq  t4, t6, 3f
  li   gp, 20
  RVTEST_FAIL
3:csrr t5, mepc
  addi t5, t5, 4
  csrw mepc, t5
  mret

  .data
RVTEST_DATA_BEGIN
.align 4
region_start:
  .zero 4096
region_end:
.align 3
counts:
  .word 0,0,0,0,0,0,0,0
causes:
  .word 0,0,0,0,0,0,0,0
  TEST_DATA
RVTEST_DATA_END
