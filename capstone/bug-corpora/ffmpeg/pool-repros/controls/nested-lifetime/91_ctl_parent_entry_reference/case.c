/* CONTROL 1, parent stays usable (pre-registration 2026-10-11-nested-lifetime-controls.md).
 *
 * A = av_buffer_pool_get(pool); av_buffer_unref(&A) returns the entry to the pool. Then the
 * entry's storage is read and written through the POOL'S OWN reference to it:
 *
 *   entry->data   BufferPoolEntry.data (libavutil/buffer_internal.h). pool_alloc_buffer stores
 *                 the allocation's own pointer there (buf->data = ret->buffer->data, i.e. what
 *                 av_buffer_alloc -> av_malloc returned); buffer_pool_flush frees the storage
 *                 through it when the pool ends. Patch 0003 keeps it unchanged ("the entry keeps
 *                 its original pointer"), derives the entry's lifetime from it
 *                 (sublet_life = CDERIVE(entry->data)) and hands each user a child of that
 *                 (sublet_issued = CDERIVE(sublet_life)), which pool_release_buffer revokes. So
 *                 entry->data is the root of the subtree the release cuts a leaf off, and the
 *                 reference the pool itself uses to manage the storage.
 *   entry->sublet_life   (protected arm only) the entry's lifetime: the direct parent of every
 *                 handed-out child, the capability CREVOKE names as the parent. Accessed as well,
 *                 after entry->data, because it is the immediate parent of the revoked child.
 *
 * The read must return what the pool left there: it does not clear an entry on return, so the
 * bytes A wrote (0x5A). The write (0xC3) must land and read back. Expected on both arms:
 * completes. A final hand-out C must be the same entry and show the parent's write, which ties
 * the parent reference to the storage the pool hands out.
 */
#include "ctl.h"

#define CTL_FILE "91_ctl_parent_entry_reference/case.c"

/* Labelled probes: each one's data accesses are the access being controlled. */
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

CTL_CASE(91, "parent") {
  CHECK(!stale, 911, "control 91 has no stale run");
  AVBufferRef *A = av_buffer_pool_get(g_pool);
  CHECK(A, 912, "av_buffer_pool_get returned NULL");
  BufferPoolEntry *entry = A->buffer->opaque; /* pool_release_buffer's opaque: the entry */
  CHECK(entry && entry->pool == g_pool, 913, "A's AVBuffer opaque is not an entry of g_pool");
  ctl_ptr("A.data", A->data);
  ctl_ptr("entry.data", entry->data);
#if CTL_PROTECTED
  ctl_ptr("entry.sublet_life", entry->sublet_life);
  ctl_ptr("entry.sublet_issued", entry->sublet_issued);
#endif
  CHECK(CTL_CURSOR(A->data) == CTL_CURSOR(entry->data), 914,
        "A->data and entry->data name different addresses");
  memset(A->data, 0x5A, POOL_BYTES);

  av_buffer_unref(&A); /* pool_release_buffer: back to the pool; protected: CREVOKE the child */
  printf("CTL-STEP released A to the pool\n");

  unsigned bad;
  CTL_ANNOUNCE("parent-read", "read", "entry->data");
  bad = read_probe(entry->data, POOL_BYTES, 0x5A);
  printf("CTL-ACCESS-DONE id=parent-read mismatches=%u expect=0x5a\n", bad);
  CHECK(bad == 0, 915, "the pool's own reference read %u bytes other than A's 0x5A", bad);

  CTL_ANNOUNCE("parent-write", "write", "entry->data");
  write_probe(entry->data, POOL_BYTES, 0xC3);
  printf("CTL-ACCESS-DONE id=parent-write value=0xc3\n");

  CTL_ANNOUNCE("parent-readback", "read", "entry->data");
  bad = read_probe(entry->data, POOL_BYTES, 0xC3);
  printf("CTL-ACCESS-DONE id=parent-readback mismatches=%u expect=0xc3\n", bad);
  CHECK(bad == 0, 916, "the parent's write did not read back (%u bytes)", bad);

#if CTL_PROTECTED
  CTL_ANNOUNCE("lifetime-read", "read", "entry->sublet_life");
  bad = read_probe(entry->sublet_life, POOL_BYTES, 0xC3);
  printf("CTL-ACCESS-DONE id=lifetime-read mismatches=%u expect=0xc3\n", bad);
  CHECK(bad == 0, 917, "the entry's lifetime read %u bytes other than 0xC3", bad);

  CTL_ANNOUNCE("lifetime-write", "write", "entry->sublet_life");
  write_probe(entry->sublet_life, POOL_BYTES, 0x3C);
  printf("CTL-ACCESS-DONE id=lifetime-write value=0x3c\n");
  CTL_ANNOUNCE("lifetime-readback", "read", "entry->data");
  bad = read_probe(entry->data, POOL_BYTES, 0x3C);
  printf("CTL-ACCESS-DONE id=lifetime-readback mismatches=%u expect=0x3c\n", bad);
  CHECK(bad == 0, 918, "the lifetime's write is not visible through entry->data (%u bytes)", bad);
  const unsigned char last = 0x3C;
#else
  const unsigned char last = 0xC3;
#endif

  /* The pool still hands the entry out, and the hand-out sees what the parent wrote. */
  AVBufferRef *C = av_buffer_pool_get(g_pool);
  CHECK(C, 919, "av_buffer_pool_get after the parent accesses returned NULL");
  ctl_ptr("C.data", C->data);
  CHECK(C->buffer->opaque == entry, 920, "the pool handed out another entry");
  CTL_ANNOUNCE("successor-read", "read", "C->data");
  bad = read_probe(C->data, POOL_BYTES, last);
  printf("CTL-ACCESS-DONE id=successor-read mismatches=%u expect=0x%02x\n", bad, last);
  CHECK(bad == 0, 921, "C does not see the parent's write (%u bytes)", bad);
  av_buffer_unref(&C);
  printf("CTL-CHECK parent ok\n");
  return 0;
}
