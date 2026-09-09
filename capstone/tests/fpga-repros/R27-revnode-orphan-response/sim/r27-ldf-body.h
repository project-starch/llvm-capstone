/* R-27 candidate, the load-fault arm (third exception shape, 2026-09-09). The ecall and misaligned-load shapes put
 * the exception on an instruction whose fault is known at decode/issue, and the waveform showed no younger LDC is
 * dispatched before such an exception commits. A load whose fault is raised inside the load unit after issue is
 * what real code produces: it issues, the next instruction issues behind it, and the fault commits two or three
 * cycles later -- inside the younger LDC's revocation-query window. Shape: a locked PMP entry (L=1, no R/W/X) over
 * one 64-byte line so that an M-mode `ld` from it takes LD_ACCESS_FAULT (cause 5) at the PMP check; the arm is
 * that `ld`, N nops, eight LDCs. The handler counts the trap and steps over the load; the chain re-executes.
 * Written before the first run: on the unmodified tree some small N hang; the control (a legal load) completes
 * at every N. Codes: PASS; 11 wrong LDC value; 12 wrong trap count; 20 unexpected cause; TIMEOUT = the hang.
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
  lla  s11, forbidden
  lla  s10, miss_line
  # locked NAPOT entry over the 64-byte `forbidden` line, no permissions: applies to M-mode too
  srli t0, s11, 2
  ori  t0, t0, 0x7
  csrw pmpaddr0, t0
  li   t0, 0x98
  csrw pmpcfg0, t0
  li   s8, 0
  .rept 100
  nop
  .endr
  li   gp, 11
  # ---- the arm
  R27_FAULT
  .rept R27_N
  nop
  .endr
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
  li   t1, 5
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
forbidden:
  .dword 0x7777777777777777
  .zero 56
.align 6
miss_line:
  .dword 0x6666666666666666
  .zero 56
  TEST_DATA
RVTEST_DATA_END
