/* Case 0: 1c7a70c9d9 -- a bucket carries the wrong connection's allocator
 *
 * Shape: wrong allocator / destroy / free through a dead list
 * Consumer: modules/http2/mod_proxy_http2.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

APRB_CASE(0) {
  /* Two connections, each with its own bucket allocator, as httpd gives every
   * conn_rec one. */
  apr_pool_t *backend_pool = NULL, *frontend_pool = NULL;
  CHECK(apr_pool_create(&backend_pool, root) == APR_SUCCESS, 710);
  CHECK(apr_pool_create(&frontend_pool, root) == APR_SUCCESS, 711);
  apr_bucket_alloc_t *backend = apr_bucket_alloc_create(backend_pool);
  apr_bucket_alloc_t *frontend = apr_bucket_alloc_create(frontend_pool);
  CHECK(backend && frontend, 712);

  struct brigade out = {.pool = frontend_pool};

  /* The defect: data for the FRONTEND connection is allocated with the
   * BACKEND's allocator. The fix uses the frontend's. */
  apr_bucket_alloc_t *chosen = fixed ? frontend : backend;
  void *data = apr_bucket_alloc(64, chosen);
  CHECK(data, 713);
  memset(data, 0xA1, 64);
  out.bucket[out.n] = data;
  out.from[out.n] = chosen;
  out.n++;

  unsigned long before = freed_to_malloc;
  /* The backend connection is done and its allocator goes away. */
  apr_bucket_alloc_destroy(backend);
  apr_pool_destroy(backend_pool);

  /* The frontend now returns what it thinks is its own storage. apr_bucket_free
   * reads node->alloc, which is the list the destroy just handed back. */
  int through_dead_list = out.from[0] != frontend;
  unsigned long freed = freed_to_malloc - before;

  printf("allocator_used=%s outlived_by_holder=%d freed_to_malloc=%lu\n",
         through_dead_list ? "backend" : "frontend", through_dead_list, freed);
  APRB_VERDICT(!fixed && through_dead_list, fixed && !through_dead_list,
               "the frontend holds storage whose allocator has been destroyed",
               "the storage came from the allocator that outlives it");
  if (!through_dead_list)
    apr_bucket_free(out.bucket[0]); /* only safe in the fixed arm */
  apr_bucket_alloc_destroy(frontend);
  return !fixed ? !through_dead_list : !!through_dead_list;
}
