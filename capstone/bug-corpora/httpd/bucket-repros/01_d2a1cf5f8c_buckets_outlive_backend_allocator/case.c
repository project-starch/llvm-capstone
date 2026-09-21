/* Case 1: d2a1cf5f8c -- buffered buckets outlive the allocator that made them
 *
 * Shape: not flushed before the allocator goes / blocks returned / stale hold
 * Consumer: modules/proxy/mod_proxy_http.c and the network filters
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

APRB_CASE(1) {
  o->defect_text = "buckets stay buffered after their allocator's blocks went back";
  o->fixed_text = "the buckets were flushed before the allocator was destroyed";
  apr_pool_t *conn_pool = NULL;
  CHECK(apr_pool_create(&conn_pool, root) == APR_SUCCESS, 720);
  apr_allocator_t *shared = apr_pool_allocator_get(conn_pool);
  apr_bucket_alloc_t *backend = apr_bucket_alloc_create_ex(shared);
  CHECK(backend, 721);

  /* Data left buffered in the network filters, made with the backend
   * connection's allocator. */
  unsigned char *buffered = apr_bucket_alloc(64, backend);
  CHECK(buffered, 722);
  memset(buffered, 0xA1, 64);
  struct brigade kept = {.pool = conn_pool};
  kept.bucket[kept.n] = buffered;
  kept.from[kept.n] = backend;
  kept.n++;

  /* The fix flushes -- the buckets go out and are freed -- before the backend
   * connection's allocator is destroyed. The defect leaves them buffered. */
  if (fixed) {
    apr_bucket_free(buffered);
    kept.n = 0;
  }

  /* conn->close, or the worker address is not reusable: the allocator goes,
   * and its blocks are on APR's free list for the next taker. */
  apr_bucket_alloc_destroy(backend);
  apr_pool_t *next = NULL;
  CHECK(apr_pool_create(&next, root) == APR_SUCCESS, 723);
  unsigned char *reissued = apr_palloc(next, 64);
  CHECK(reissued, 724);
  memset(reissued, 0xB2, 64);
  o->still_held = kept.n > 0;
  if (o->still_held) {
    /* The filter chain sends what it still holds: the first read of the
     * buffered bucket is the probe. */
    held = buffered;
    mark(1);
    o->now = read_probe(held);
  }
  o->defect = o->still_held;
  o->held_up = !o->still_held;
}
