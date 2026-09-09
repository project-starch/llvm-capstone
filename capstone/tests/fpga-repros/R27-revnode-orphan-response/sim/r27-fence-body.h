/* R-27 candidate (found 2026-09-09 while validating the R-26 flush): a pipeline flush that lands while a
 * capability instruction's revocation-node validity query is in flight orphans the node's response. The
 * DYN unit is flushed (ex_stage.sv wires flush_i to its flush endpoint), the revocation node is not, and the
 * node then blocks in `send ep.query_res` (capstone_rev_node.anvil) until someone acks a response nobody
 * asked for; the next capability instruction that queries (LDC/STC/CALL/RETURN/REVOKE/SPLIT/TIGHTEN/LCC)
 * waits forever and the pipeline deadlocks behind it.
 * Arm shape: a store (drains through the store buffer for ~S12_MEM_DELAY cycles), a fence (commits only
 * when the store buffer is empty, then flushes IF/ID/EX), then a chain of LDCs that issue and query the node
 * while the fence waits. Written before the first run: the fence flush lands inside one LDC's query window
 * and the re-issued chain hangs; with the store drained before the fence, without the store, or without the
 * fence the chain completes. READ 2026-09-09 (unmodified tree): the prediction of WHICH arm hangs was wrong --
 * `fence-st` (an explicit store) completed and `fence-nost` HUNG (the setup's own STC was still draining, and its
 * fence commit fell three cycles after the first LDC's dispatch); with the drain fix both complete. Codes: PASS = every LDC returned the stored capability; 11 = a wrong value;
 * TIMEOUT = the hang. Macros: R27_OLDER, R27_FENCE, R27_PAD.
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
  li   t6, 0x5a5a
  li   gp, 11
  # ---- the arm
  R27_OLDER
  R27_FENCE
  R27_PAD
  LDC_CHECK(t1)
  LDC_CHECK(t2)
  LDC_CHECK(t3)
  LDC_CHECK(t4)
  LDC_CHECK(t5)
  LDC_CHECK(s2)
  LDC_CHECK(s3)
  LDC_CHECK(s4)
  RVTEST_PASS
fail:
  RVTEST_FAIL
.align 2
m_trap_handler:
  li   gp, 20
  RVTEST_FAIL
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
