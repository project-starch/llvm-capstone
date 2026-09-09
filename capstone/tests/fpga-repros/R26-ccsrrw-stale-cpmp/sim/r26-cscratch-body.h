/* R-26, the switcher arm. Included by r26-v2-cscratch{,-fence}.S which define R26_DELAY / R26_OLDER / R26_FENCE.
 * Question: after CCSRRW writes a NEW capability to CSCRATCH, does a CALL that follows at once (with an older
 * cache-missing load ahead of the CSR write) save the NEW cscratch into the caller's context, or the OLD one?
 * Prediction (written 2026-09-09 before the first run): the NEW one, before and after the R-26 flush fix — the
 * domain switcher reads CSRs only after it accepts the CALL's commit handshake, commit is in order, so the
 * older CCSRRW has always committed. This arm is a no-regression guard for CALL/RETURN under the new flush.
 * Shape follows call-hot-cache.S. Verdict: after RETURN the caller reads CSCRATCH back (destructive read, the
 * last access) and compares its cursor with the NEW capability's; the callee checks it received the context
 * slot's capability. Exit codes: PASS; 11 = the OLD value came back (hazard); 12 = the callee saw the wrong
 * cscratch; 20 = an unexpected trap.
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
  lla  a1, _stub_end
  CAPENTER(a0, a1)

  # OLD cscratch: NONLIN over old_region
  MKNONLIN(s8, old_region, old_region_end)
  CCSRRW(x0, CCSR_CSCRATCH, s8)
  # NEW cscratch, written in the arm: NONLIN over new_region; s7 = its cursor for the compare
  MKNONLIN(s6, new_region, new_region_end)
  lla  s7, new_region
  # the capability the callee must see as its cscratch: NONLIN over ctx_region
  MKNONLIN(s9, ctx_region, ctx_region_end)
  lla  s10, ctx_region
  # the older load's target: an untouched line
  lla  s11, miss_line

  # ctvec capability (as call-hot-cache.S)
  CAPCREATE(a7)
  li   a0, CAP_TYPE_LIN
  li   a2, 0x80001000
  li   a3, 0x90000000
  li   a4, CAP_PERM_RWX
  CAPTYPE(a7, a0)
  CAPBOUND(a7, a2, a3)
  CAPPERM(a7, a4)
  # context region capability a1 (LIN RWX)
  CAPCREATE(a1)
  li   a0, CAP_TYPE_LIN
  lla  a2, context_start
  lla  a3, context_end
  li   a4, CAP_PERM_RWX
  CAPTYPE(a1, a0)
  CAPBOUND(a1, a2, a3)
  CAPPERM(a1, a4)
  # callee pc capability a2 (LIN RWX over the stub)
  CAPCREATE(a2)
  li   a0, CAP_TYPE_LIN
  lla  a3, _stub
  lla  a4, _stub_end
  li   a5, CAP_PERM_RWX
  CAPTYPE(a2, a0)
  CAPBOUND(a2, a3, a4)
  CAPPERM(a2, a5)
  # context slots
  li   a0, 0x0000000A00040000
  STC(a1, a2, 0)          # pc
  STC(a1, a7, 16)         # ctvec
  STC(a1, s9, 32)         # cscratch the callee receives
  sd   a0, 48(a1)         # mstatus
  sd   a3, 56(a1)         # mideleg
  sd   a4, 64(a1)         # medeleg
  sd   a5, 72(a1)         # mip
  sd   a6, 80(a1)         # mie
  SEAL(a3, a1)

  # ---- the arm
  R26_DELAY
  R26_OLDER
#ifdef R26_WRITE
  R26_WRITE
#else
  CCSRRW(x0, CCSR_CSCRATCH, s6)
#endif
  R26_FENCE
  CALL(a6, a3)

  # ---- back in the caller: the switcher must have saved the NEW cscratch and restored it here
  CCSRRW(t0, CCSR_CSCRATCH, x0)   # destructive read; last access
  CAPPRINT(t0)
  li   gp, 11
#ifdef R26_WRONG_EXPECT
  bne  t0, s10, fail          # positive control: compare against the WRONG capability; must FAIL 11
#elif defined(R26_EXPECT_OLD)
  lla  t2, old_region         # the arm did not write CSCRATCH: the OLD capability must come back
  bne  t0, t2, fail
#else
  bne  t0, s7, fail
#endif
  RVTEST_PASS

_stub:
  CCSRRW(t1, CCSR_CSCRATCH, x0)   # what the switch installed for the callee
  CAPPRINT(t1)
  li   gp, 12
  bne  t1, s10, fail
  la   a1, _stub
  RETURN(ra, a1, x0)
_stub_end:

fail:
  RVTEST_FAIL

.align 4
m_trap_handler:
  li   gp, 20
  RVTEST_FAIL

  .data
RVTEST_DATA_BEGIN
.align 6
context_start:
  .zero 4096*4
context_end:
.align 6
old_region:
  .zero 64
old_region_end:
.align 6
new_region:
  .zero 64
new_region_end:
.align 6
ctx_region:
  .zero 64
ctx_region_end:
.align 6
miss_line:
  .dword 0x6666666666666666
  .zero 56
  TEST_DATA
RVTEST_DATA_END
