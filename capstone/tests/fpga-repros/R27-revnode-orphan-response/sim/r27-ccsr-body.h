/* R-27 x R-26 (the shipping question): after the R-26 fix a CCSRRW to CPMP/CSCRATCH/CEPC flushes IF/ID/EX at
 * commit; if a younger LDC has already sent its revocation query, the flush orphans it (see r27-fence-body.h).
 * Arm: CCSRRW CPMP0 <- a NONLIN RW capability (harmless in M-mode without MPRV), R27_PAD nops, the LDC chain.
 * Written before the first run (2026-09-09): on the UNMODIFIED tree every N completes (CCSRRW does not flush);
 * on the R-26-fixed tree the small N hang and the large N complete. This arm decides whether the R-26 flush
 * can ship without the R-27 fix. The `-older` arms put a cache-missing load ahead of the CCSRRW, which delays its
 * commit (and so its flush) by the memory latency; the nop count then walks an LDC's dispatch across that later
 * flush tick — prediction: on the fixed tree one or two N hang, on the unmodified tree all complete.
 *
 * (body copied from r27-fence-body.h; original header follows)
 * R-27 candidate (found 2026-09-09 while validating the R-26 flush): a pipeline flush that lands while a
 * capability instruction's revocation-node validity query is in flight orphans the node's response. The
 * DYN unit is flushed (ex_stage.sv wires flush_i to its flush endpoint), the revocation node is not, and the
 * node then blocks in `send ep.query_res` (capstone_rev_node.anvil) until someone acks a response nobody
 * asked for; the next capability instruction that queries (LDC/STC/CALL/RETURN/REVOKE/SPLIT/TIGHTEN/LCC)
 * waits forever and the pipeline deadlocks behind it.
 * Arm shape: a store (drains through the store buffer for ~S12_MEM_DELAY cycles), a fence (commits only
 * when the store buffer is empty, then flushes IF/ID/EX), then a chain of LDCs that issue and query the node
 * while the fence waits. Written before the first run: the fence flush lands inside one LDC's query window
 * and the re-issued chain hangs; with the store drained before the fence, without the store, or without the
 * fence the chain completes. Codes: PASS = every LDC returned the stored capability; 11 = a wrong value;
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
  MKNONLIN(s10, region, region_end)
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
