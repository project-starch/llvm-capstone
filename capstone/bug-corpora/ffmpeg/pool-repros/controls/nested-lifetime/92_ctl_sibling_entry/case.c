/* CONTROL 2, sibling stays usable (pre-registration 2026-10-11-nested-lifetime-controls.md).
 *
 * A and B are two hand-outs from the same AVBufferPool, held at once, so they are two entries
 * (pool_alloc_buffer twice). A is filled with 0xA1 and B with 0xB2; av_buffer_unref(&A) returns A's
 * entry to the pool, which on the protected arm revokes A's child (pool_release_buffer, CREVOKE).
 * Then B is read through B->data, its own reference: its contents must be intact (0xB2), and a
 * write (0xB7) must land and read back. Expected on both arms: completes, intact.
 */
#include "ctl.h"

#define CTL_FILE "92_ctl_sibling_entry/case.c"

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

CTL_CASE(92, "sibling") {
  CHECK(!stale, 921, "control 92 has no stale run");
  AVBufferRef *A = av_buffer_pool_get(g_pool);
  AVBufferRef *B = av_buffer_pool_get(g_pool);
  CHECK(A && B, 922, "av_buffer_pool_get returned NULL");
  CHECK(A->buffer->opaque != B->buffer->opaque, 923, "A and B are the same entry");
  ctl_ptr("A.data", A->data);
  ctl_ptr("B.data", B->data);
  memset(A->data, 0xA1, POOL_BYTES);
  memset(B->data, 0xB2, POOL_BYTES);

  av_buffer_unref(&A); /* A's entry back to the pool; protected: CREVOKE A's child */
  printf("CTL-STEP released A to the pool\n");

  unsigned bad;
  CTL_ANNOUNCE("sibling-read", "read", "B->data");
  bad = read_probe(B->data, POOL_BYTES, 0xB2);
  printf("CTL-ACCESS-DONE id=sibling-read mismatches=%u expect=0xb2\n", bad);
  CHECK(bad == 0, 924, "B is not intact after A's release: %u bytes other than 0xB2", bad);

  CTL_ANNOUNCE("sibling-write", "write", "B->data");
  write_probe(B->data, POOL_BYTES, 0xB7);
  printf("CTL-ACCESS-DONE id=sibling-write value=0xb7\n");

  CTL_ANNOUNCE("sibling-readback", "read", "B->data");
  bad = read_probe(B->data, POOL_BYTES, 0xB7);
  printf("CTL-ACCESS-DONE id=sibling-readback mismatches=%u expect=0xb7\n", bad);
  CHECK(bad == 0, 925, "B's write did not read back (%u bytes)", bad);

  av_buffer_unref(&B);
  printf("CTL-CHECK sibling ok, B intact\n");
  return 0;
}
