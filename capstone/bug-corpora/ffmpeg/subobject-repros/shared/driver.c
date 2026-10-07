/* main(), the arenas and the pool. One program per case, as the contract says:
 * a capability fault ends a domain, so a case that provokes one cannot also
 * report results beside it. */
#include "corpus.h"
#ifdef FFPOOL_BORROW_LINEAR
#include "payload-backend.h"
#endif

AVBufferPool *g_pool;
AVRefStructPool *g_refpool;

_Noreturn void ff2_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}
void ff2_sink(const struct ff2_event *event) { (void)event; }
void ff2_lock(void) {}
void ff2_unlock(int *guard) { (void)guard; }

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != ff2_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            ff2_case_number, argv[2]);
    return 75;
  }
  /* An optional FOURTH word selects what the payload backend enforces, and the
   * three values are the backend's own (src/capstone-domain/payload-
   * capabilities.c), separated by the GRANULARITY of the revoke:
   *
   *   0  per-object bounds only: one alias per block, shrunk to each lease's
   *      requested size. Nothing is ever revoked. The only shape a native
   *      backend can run, which is why the native arms pass 0.
   *   1  the above plus an outer handle per block, revoked when the BLOCK goes
   *      back. A lease still hands out the block's own alias.
   *   2  fresh authority minted for each lease and revoked when the LEASE goes
   *      back, which is what a pool recycling a buffer does.
   *
   * Absent means 0, so every existing invocation is unchanged. The arms are
   * therefore one binary differing in one argument, which is what the corpus
   * contract asks of paired arms. */
  unsigned mode = argc > 3 ? (unsigned)atoi(argv[3]) : 0;
  if (mode > 2) {
    fprintf(stderr, "CONTROL-FAILED mode %s is not 0, 1 or 2\n", argv[3]);
    return 75;
  }
  void *metadata = aligned_alloc(64, FF2_META_BYTES);
  if (!metadata)
    ff2_fail(604);
  ff2_memory_init(metadata, FF2_META_BYTES);
#ifdef FFPOOL_BORROW_LINEAR
  /* The metadata above stays an ordinary heap object: it is the side table the
   * backend needs AFTER a revoke, so it must not live in the revoked region. */
  ff2_payload_borrow(FF2_PAYLOAD_BYTES);
#else
  void *payload = aligned_alloc(64, FF2_PAYLOAD_BYTES);
  if (!payload)
    ff2_fail(604);
  ff2_payload_init(payload, FF2_PAYLOAD_BYTES);
  if (mode)
    ff2_fail(607); /* no authority to revoke without a linear payload */
#endif
  ff2_set_mode(mode);
  ff2_reset();
  g_pool = av_buffer_pool_init(POOL_BYTES, NULL);
  if (!g_pool)
    ff2_fail(605);
  /* The side-table pool. Created unconditionally so every case links the same
   * driver; cases that do not use it are unaffected, which the re-run of cases
   * 0-2 against this driver is there to show. */
  g_refpool = av_refstruct_pool_alloc(TAB_BYTES, 0);
  if (!g_refpool)
    ff2_fail(606);
  printf("case=%d arm=%s\n", ff2_case_number, fixed ? "fixed" : "buggy");
  int rc = ff2_case_run(fixed);
  av_buffer_pool_uninit(&g_pool);
  av_refstruct_pool_uninit(&g_refpool);
  free(metadata);
#ifndef FFPOOL_BORROW_LINEAR
  free(payload);
#endif
  return rc;
}
