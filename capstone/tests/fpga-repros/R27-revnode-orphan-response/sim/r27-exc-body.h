/* R-27 candidate, the exception arm: does a flush raised by an OLDER instruction's exception orphan a younger
 * capability instruction's revocation-node query the same way the fence flush does? Shape: a 64-cycle divide
 * holds commit; N nops; an `ecall` (its exception is known at decode but commits only behind the divide and the
 * nops; the first version used a misaligned `ld`, which did not trap here and returned code 12 at every N); a CSR read that cannot issue until the divider frees the fixed-latency unit, so the LDC chain
 * behind it issues in the cycles around the divide's commit; then eight LDCs. The nop count moves the
 * exception's commit (the flush) across the first LDC's query window a cycle or two at a time. The trap
 * handler counts the ecall trap (cause 11) and steps over it, so the chain re-executes after mret.
 * Written before the first run (2026-09-09): on the unmodified tree some nop counts hang (flush inside a
 * query window), N = 0 completes (flush before the first query), and the control with a legal load
 * completes at every N. Codes: PASS; 11 = an LDC returned a wrong value; 12 = wrong trap count; 20 = a
 * trap with an unexpected cause; TIMEOUT = the hang. Macros: R27_N (nop count), R27_FAULT (the load).
 */
#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"
#define MKNONLIN(CREG, LO, HI) \
  CAPCREATE(CREG); li a0, CAP_TYPE_NONLIN; lla a2, LO; lla a3, HI; li a4, CAP_PERM_RW; \
  CAPTYPE(CREG, a0); CAPBOUND(CREG, a2, a3); CAPPERM(CREG, a4);
#define LDC_CHECK(RD) LDC(RD, s6, 0); bne RD, s9, fail
.section .text.init
.globl _start
_start:
  la   t0, m_trap_handler
  csrw mtvec, t0
  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)
  MKNONLIN(s6, region, region_end)
  MKNONLIN(s7, payload, payload_end)
  lla  s9, payload
  STC(s6, s7, 0)
  lla  s11, miss_line
  li   s8, 0
  li   a5, 1
  li   a6, 3
  .rept 100
  nop
  .endr
  li   gp, 11
  # ---- the arm
  div  a5, a5, a6
  .rept R27_N
  nop
  .endr
  R27_FAULT
  csrr t2, mcycle
  LDC_CHECK(t1)
  LDC_CHECK(t3)
  LDC_CHECK(t4)
  LDC_CHECK(t5)
  LDC_CHECK(s2)
  LDC_CHECK(s3)
  LDC_CHECK(s4)
  LDC_CHECK(s5)
  li   gp, 12
  li   t0, R27_TRAPS
  bne  s8, t0, fail
  RVTEST_PASS
fail:
  RVTEST_FAIL
.align 2
m_trap_handler:
  csrr t0, mcause
  li   t1, 11
  li   gp, 20
  bne  t0, t1, fail
  addi s8, s8, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
_end_of_text:
  .data
RVTEST_DATA_BEGIN
.align 6
region:
  .zero 64
region_end:
.align 6
payload:
  .zero 64
payload_end:
.align 6
miss_line:
  .dword 0x6666666666666666
  .zero 56
  TEST_DATA
RVTEST_DATA_END
