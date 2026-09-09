/* s06agg-shape: the STRUCT-ASSIGNMENT shape of S-06, as a directed test.
 *
 * The board rung s06agg (tests/fpga-repros/S06-.../src/) returned 66 on 2026-09-09 (boot sw46) --
 * y lost, x intact -- and that draw has no earlier recorded run on any bitstream, so the board
 * cannot say whether the shape regressed or was never fixed. This test answers that off the board:
 * run it on ef5a8eaf2 and on 66c4e7517 and compare.
 *
 * The shape, which untagged-ldc-stc-128.S does NOT create: a 32-byte object whose FIRST 16-byte
 * chunk holds a REAL CAPABILITY and whose SECOND chunk at offset 0x10 holds two plain u64s. The
 * compiler lowers `*d = *s` to two capability-grained copies, and the second moves x and y together
 * through one untagged 128-bit round trip with a tagged chunk immediately before it. The existing
 * test round-trips offsets 0 and 32 with no adjacent capability, so it cannot see any interaction
 * between the tag of chunk 0 and the data of chunk 1.
 *
 * Codes: PASS both halves survived; 11 = HIGH half (y) lost with x intact -- S-06's signature, the
 * same reading the rung's 66 encodes; 12 = LOW half (x) lost; 13 = both lost; 14 = the plain sd/ld
 * control is wrong, so the run carries NO verdict about ldc/stc; 20 = unexpected trap.
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

  MKNONLIN(s1, src_region, src_region_end)
  MKNONLIN(s2, dst_region, dst_region_end)
  MKNONLIN(s3, payload, payload_end)

  li   s5, 0x1111222233334444
  sd   s5, 16(s1)                 # src.x, LOW half of chunk 1 -- stored EARLY
  li   s7, 0x0f0f0f0f0f0f0f0f
  sd   s7, 48(s1)                 # control, never touched by ldc/stc
  STC(s1, s3, 0)                  # src chunk 0 = a REAL capability

  # ---- the construct, in the committed rung's INSTRUCTION ORDER (s06agg.dom 249118220f8c at
  # 0x10438): the `sd` of y is the instruction IMMEDIATELY before the `ldc` of that line, with
  # nothing between, while x was stored several instructions earlier with its constant build in
  # between; and the PLAIN chunk is copied first, the pointer chunk after. Adjacency is the point:
  # the half the rung lost is the half whose store had not left the write buffer, which makes this
  # a store-to-load forwarding shape as well as a tag-path one. Written apart, the test contains
  # the shape without creating the condition.
  li   s6, 0x5555666677778888     # constant build, as the rung has it
  sd   s6, 24(s1)                 # src.y, HIGH half -- the LAST plain store to the line
  LDC(t4, s1, 16)                 # immediately behind it: the 16-byte line, x and y together
  STC(s2, t4, 16)
  LDC(t3, s1, 0)                  # the pointer chunk, after
  STC(s2, t3, 0)

  ld   t5, 16(s2)                 # x back
  ld   t6, 24(s2)                 # y back
  ld   s8, 48(s1)                 # control back
  CAPPRINT(t5)
  CAPPRINT(t6)
  CAPPRINT(s8)

  li   gp, 14
  bne  s8, s7, fail               # control first: a wrong control voids the run
  li   gp, 13
  bne  t5, s5, 1f
  bne  t6, s6, 2f
  RVTEST_PASS
1:                                # x wrong
  li   gp, 13
  bne  t6, s6, fail               # both wrong
  li   gp, 12
  j    fail
2:                                # x right, y wrong -- S-06's signature
  li   gp, 11
  j    fail

fail:
  RVTEST_FAIL

.align 4
m_trap_handler:
  li   gp, 20
  RVTEST_FAIL
_end_of_text:

  .data
RVTEST_DATA_BEGIN
.align 6
src_region:
  .zero 128
src_region_end:
.align 6
dst_region:
  .zero 128
dst_region_end:
.align 6
payload:
  .zero 64
payload_end:
  TEST_DATA
RVTEST_DATA_END
