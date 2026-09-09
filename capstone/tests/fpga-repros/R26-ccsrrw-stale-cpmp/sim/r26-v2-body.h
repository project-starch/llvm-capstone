/* R-26 directed test body. Included by r26-cpmp-{nodelay,div,fence}.S which define R26_DELAY / R26_FENCE.
 * Question: after CCSRRW narrows CPMP[0], does the very next (younger) load run its CPMP check against the
 * OLD entry? All in M-mode with mstatus.MPRV=1/MPP=S, so M-mode loads are CPMP-checked (pmp_data_if uses
 * ld_st_priv_lvl) and NO mret/flush sits between the CSR write and the load.
 * Phases (counts[phase] = LD_ACCESS_FAULT traps in that phase):
 *   0  positive control: checked load OUTSIDE the wide entry -> must trap (proves the check fires)
 *   1  THE ARM: [delay] CCSRRW CPMP0 <- narrow; [fence.i]; load in wide but outside narrow
 *        trap     -> the load saw the new entry (no hazard here)
 *        no trap  -> the load ran against the stale entry (R-26 hazard)
 *   2  after an explicit fence.i the same load -> must trap (proves the write landed)
 * Result: PASS = counts 1/1/1.  FAIL codes (tohost): 10 control missing, 11 = HAZARD (arm did not trap),
 * 12 post-fence load did not trap, 20 unexpected trap cause.
 */
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define ARM_S  li t0, MSTATUS_MPP; csrc mstatus, t0; li t0, (MSTATUS_MPRV | (1 << 11)); csrs mstatus, t0
#define DISARM li t0, MSTATUS_MPRV; csrc mstatus, t0

.section .text.init
.globl _start
_start:
  la   t0, counts
  sw   x0, 0(t0)
  sw   x0, 4(t0)
  sw   x0, 8(t0)
  la   t0, phase
  sw   x0, 0(t0)
  la   a0, _start
  la   a1, _end_of_text
  CAPENTER(a0, a1)
  li   t0, 0x001fffffffffffff
  csrw pmpaddr0, t0
  li   t0, 0x1f
  csrw pmpcfg0, t0
  # wide NONLIN RW capability over [wide_lo, wide_hi) -> CPMP[0]
  CAPCREATE(a1)
  li   a0, CAP_TYPE_NONLIN
  .insn r 0x7B, 0x0, 0x5, a1, a0, a1
  li   a0, CAP_PERM_RW
  .insn r 0x7B, 0x0, 0x7, a1, a0, a1
  la   a2, wide_lo
  la   a3, wide_hi
  CAPBOUND(a1, a2, a3)
  CCSRRW(x0, CCSR_CPMP(0), a1)
  # narrow NONLIN RW capability over [wide_lo, narrow_hi), kept in s1 for the arm
  CAPCREATE(s1)
  li   a0, CAP_TYPE_NONLIN
  .insn r 0x7B, 0x0, 0x5, s1, a0, s1
  li   a0, CAP_PERM_RW
  .insn r 0x7B, 0x0, 0x7, s1, a0, s1
  la   a2, wide_lo
  la   a3, narrow_hi
  CAPBOUND(s1, a2, a3)
  la   t0, m_trap_handler
  csrw mtvec, t0
  li   s5, 0
  la   s2, probe_data      # inside wide, outside narrow
  la   s6, older_line      # inside narrow (allowed by both entries), untouched until the arm -> a cache miss
  la   s3, outside_data    # outside wide
  # ---- phase 0: positive control
  li   gp, 1
  ARM_S
  ld   t1, 0(s3)
  DISARM
  # ---- phase 1: the arm
  li   s5, 1
  li   gp, 2
  ARM_S
  R26_DELAY
  R26_OLDER
  CCSRRW(x0, CCSR_CPMP(0), s1)
  R26_FENCE
  ld   t1, 0(s2)
  DISARM
  # ---- phase 2: the same load after a definite flush
  li   s5, 2
  li   gp, 3
  ARM_S
  fence.i
  ld   t1, 0(s2)
  DISARM
  # ---- verify
  la   t0, counts
  lw   t1, 0(t0)
  li   t2, 1
  li   gp, 10
  bne  t1, t2, fail
  lw   t1, 8(t0)
  li   gp, 12
  bne  t1, t2, fail
  lw   t1, 4(t0)
  li   gp, 11
  bne  t1, t2, fail
  RVTEST_PASS
fail:
  RVTEST_FAIL

.align 4
m_trap_handler:
  DISARM
  csrr t0, mcause
  li   t1, 5                 # load access fault
  bne  t0, t1, unexpected
  slli t1, s5, 2
  la   t0, counts
  add  t0, t0, t1
  lw   t2, 0(t0)
  addi t2, t2, 1
  sw   t2, 0(t0)
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  li   t0, MSTATUS_MPP
  csrs mstatus, t0           # return to M-mode
  mret
unexpected:
  li   gp, 20
  RVTEST_FAIL

_end_of_text:

  .data
RVTEST_DATA_BEGIN
.align 6
wide_lo:
inside_data:
  .dword 0x1111111111111111
  .dword 0x2222222222222222
.align 6
older_line:
  .dword 0x6666666666666666
  .dword 0x7777777777777777
.align 6
narrow_hi:
probe_data:
  .dword 0x3333333333333333
  .dword 0x4444444444444444
.align 6
wide_hi:
outside_data:
  .dword 0x5555555555555555
.align 3
counts:
  .word 0, 0, 0, 0
phase:
  .word 0
  TEST_DATA
RVTEST_DATA_END
