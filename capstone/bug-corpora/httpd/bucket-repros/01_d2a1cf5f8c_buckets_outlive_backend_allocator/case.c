/* Case 1: d2a1cf5f8c -- buffered buckets outlive the allocator that made them
 *
 * Shape: not flushed before the allocator goes / blocks returned / stale hold
 * Consumer: modules/proxy/mod_proxy_http.c and the network filters
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

APRB_CASE(1) {
  apr_pool_t *conn_pool = NULL;
  CHECK(apr_pool_create(&conn_pool, root) == APR_SUCCESS, 720);
  apr_allocator_t *shared = apr_pool_allocator_get(conn_pool);
  apr_bucket_alloc_t *backend = apr_bucket_alloc_create_ex(shared);
  CHECK(backend, 721);

  /* Data left buffered in the network filters, made with the backend
   * connection's allocator. */
  void *buffered = apr_bucket_alloc(64, backend);
  CHECK(buffered, 722);
  memset(buffered, 0xA1, 64);
  struct brigade held = {.pool = conn_pool};
  held.bucket[held.n] = buffered;
  held.from[held.n] = backend;
  held.n++;

  /* The fix flushes -- the buckets go out and are freed -- before the backend
   * connection's allocator is destroyed. The defect leaves them buffered. */
  if (fixed) {
    apr_bucket_free(buffered);
    held.n = 0;
  }

  unsigned long before = freed_to_malloc;
  /* conn->close, or the worker address is not reusable: the allocator goes. */
  apr_bucket_alloc_destroy(backend);

  /* Its blocks are now on APR's free list, and the next taker gets them. */
  apr_pool_t *next = NULL;
  CHECK(apr_pool_create(&next, root) == APR_SUCCESS, 723);
  void *reissued = apr_palloc(next, 64);
  CHECK(reissued, 724);
  memset(reissued, 0xB2, 64);
  int still_held = held.n > 0;
  unsigned long freed = freed_to_malloc - before;

  printf("flushed_before_destroy=%d still_buffered=%d freed_to_malloc=%lu\n",
         fixed, still_held, freed);
  APRB_VERDICT(!fixed && still_held && freed == 0, fixed && !still_held,
               "buckets stay buffered after their allocator's blocks went back",
               "the buckets were flushed before the allocator was destroyed");
  return !fixed ? !(still_held && freed == 0) : !!still_held;
}
