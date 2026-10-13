/* CONTROL 3, successor stays usable, the stale reference does not (pre-registration
 * 2026-10-11-nested-lifetime-controls.md).
 *
 * A = av_buffer_pool_get(pool); A_data = A->data is kept; av_buffer_unref(&A) returns the entry.
 * Then av_buffer_pool_get until the pool hands out the entry at A's data address (the free list
 * is LIFO, so the first get; non-matching hand-outs, if any, are held so the loop progresses).
 * C's address must equal A's. C is written (0x77) and read back through C->data.
 *
 *   run    ends there: expected to complete on both arms.
 *   stale  the same sequence, then ONE read through A_data, the old pointer, at stale_read_probe.
 *          Protected (patch 0003): A's child was revoked at the return, so the read must fault
 *          with cause 25 at the probe. Release: it must complete and read C's data (0x77).
 */
#include "ctl.h"

#define CTL_FILE "93_ctl_successor_entry/case.c"
#define CTL_MAX_GETS 16

__attribute__((noinline)) unsigned read_probe(const volatile unsigned char *p, unsigned n,
                                              unsigned char expect) {
  unsigned bad = 0;
  for (unsigned i = 0; i < n; i++)
    bad += p[i] != expect;
  return bad;
}
__attribute__((noinline)) void write_probe(volatile unsigned char *p, unsigned n,
                                           unsigned char value) {
  for (unsigned i = 0; i < n; i++)
    p[i] = value;
}
/* The stale access: its first load is the read through A's old pointer. */
__attribute__((noinline)) unsigned stale_read_probe(const volatile unsigned char *p, unsigned n,
                                                    unsigned char expect) {
  unsigned bad = 0;
  for (unsigned i = 0; i < n; i++)
    bad += p[i] != expect;
  return bad;
}

CTL_CASE(93, "successor") {
  AVBufferRef *A = av_buffer_pool_get(g_pool);
  CHECK(A, 931, "av_buffer_pool_get returned NULL");
  unsigned char *A_data = A->data;
  const unsigned long a_addr = CTL_CURSOR(A_data);
  ctl_ptr("A.data", A_data);
  memset(A->data, 0x44, POOL_BYTES);

  av_buffer_unref(&A); /* back to the pool; protected: CREVOKE A's child */
  printf("CTL-STEP released A to the pool\n");

  AVBufferRef *held[CTL_MAX_GETS] = {0};
  AVBufferRef *C = NULL;
  int gets = 0;
  while (gets < CTL_MAX_GETS) {
    AVBufferRef *r = av_buffer_pool_get(g_pool);
    CHECK(r, 932, "av_buffer_pool_get returned NULL after %d gets", gets);
    gets++;
    if (CTL_CURSOR(r->data) == a_addr) {
      C = r;
      break;
    }
    held[gets - 1] = r;
  }
  CHECK(C, 933, "no hand-out at A's address 0x%lx in %d gets", a_addr, gets);
  ctl_ptr("C.data", C->data);
  printf("CTL-CHECK successor-address A=0x%lx C=0x%lx gets=%d %s\n", a_addr, CTL_CURSOR(C->data),
         gets, CTL_CURSOR(C->data) == a_addr ? "equal" : "DIFFERENT");

  unsigned bad;
  CTL_ANNOUNCE("successor-write", "write", "C->data");
  write_probe(C->data, POOL_BYTES, 0x77);
  printf("CTL-ACCESS-DONE id=successor-write value=0x77\n");

  CTL_ANNOUNCE("successor-read", "read", "C->data");
  bad = read_probe(C->data, POOL_BYTES, 0x77);
  printf("CTL-ACCESS-DONE id=successor-read mismatches=%u expect=0x77\n", bad);
  CHECK(bad == 0, 934, "C's write did not read back (%u bytes)", bad);

  if (stale) {
    CTL_ANNOUNCE("stale-read", "read", "A_data (A's pointer, held across the return)");
    bad = stale_read_probe(A_data, POOL_BYTES, 0x77);
    printf("CTL-ACCESS-DONE id=stale-read mismatches=%u expect=0x77 (C's data)\n", bad);
    CHECK(bad == 0, 935, "the stale read completed but read %u bytes other than C's 0x77", bad);
  }

  av_buffer_unref(&C);
  for (int i = 0; i < CTL_MAX_GETS; i++)
    if (held[i])
      av_buffer_unref(&held[i]);
  printf("CTL-CHECK successor ok\n");
  return 0;
}
