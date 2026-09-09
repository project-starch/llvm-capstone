/* The monitor's own shape (sbi_capstone.S call_into_smode / return path): an older cache-missing load,
 * CCSRRW to CSCRATCH (flushing after the R-26 fix), fence.i, then mret into S-mode. The CALL-shaped arm
 * hangs on the fixed tree; this arm asks whether the mret shape does. S-mode code ecalls back; the
 * handler verifies cscratch holds the NEW capability and passes. Codes: 11 wrong cscratch, 20 unexpected
 * trap cause, TIMEOUT = the hang.
 * 2026-09-09 (second version): the first version installed no code CPMP entry and an unaligned mtvec, so every
 * arm faulted on the first S-mode fetch and spun on an illegal instruction at the vector; those hangs were the test.
 */
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"
#define MKNONLIN(CREG, LO, HI) \
  CAPCREATE(CREG); li a0, CAP_TYPE_NONLIN; lla a2, LO; lla a3, HI; li a4, CAP_PERM_RW; \
  CAPTYPE(CREG, a0); CAPBOUND(CREG, a2, a3); CAPPERM(CREG, a4);
.section .text.init
.globl _start
_start:
  la   t0, m_trap_handler
  csrw mtvec, t0
  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)
  li   t0, 0x001fffffffffffff
  csrw pmpaddr0, t0
  li   t0, 0x1f
  csrw pmpcfg0, t0
  # code CPMP entry: an S/U-mode instruction fetch under capmode is CPMP-checked for X
  # (pmp_data_if.sv, the CAPSTONE_ITFC_EN block); without it the first S-mode fetch faults.
  CAPCREATE(a1)
  li   a0, CAP_TYPE_NONLIN
  .insn r 0x7B, 0x0, 0x5, a1, a0, a1
  li   a0, CAP_PERM_XO
  .insn r 0x7B, 0x0, 0x7, a1, a0, a1
  lla  a2, _start
  lla  a3, _end_of_text
  CAPBOUND(a1, a2, a3)
  CCSRRW(x0, CCSR_CPMP(0), a1)
  MKNONLIN(s8, old_region, old_region_end)
  CCSRRW(x0, CCSR_CSCRATCH, s8)
  MKNONLIN(s6, new_region, new_region_end)
  lla  s7, new_region
  lla  s11, miss_line
  li   t0, MSTATUS_MPP
  csrc mstatus, t0
  li   t0, (1 << 11)
  csrs mstatus, t0
  la   t0, s_code
  csrw mepc, t0
  # ---- the arm
  ld   t6, 0(s11)
  CCSRRW(x0, CCSR_CSCRATCH, s6)
  fence.i
  mret
.align 2
s_code:
  ecall
.align 2
m_trap_handler:
  csrr t0, mcause
  li   t1, 9                   # ecall from S
  li   gp, 20
  bne  t0, t1, fail
  CCSRRW(t2, CCSR_CSCRATCH, x0)
  CAPPRINT(t2)
  li   gp, 11
  bne  t2, s7, fail
  RVTEST_PASS
fail:
  RVTEST_FAIL
_end_of_text:
  .data
RVTEST_DATA_BEGIN
.align 6
old_region:
  .zero 64
old_region_end:
.align 6
new_region:
  .zero 64
new_region_end:
.align 6
miss_line:
  .dword 0x6666666666666666
  .zero 56
  TEST_DATA
RVTEST_DATA_END
