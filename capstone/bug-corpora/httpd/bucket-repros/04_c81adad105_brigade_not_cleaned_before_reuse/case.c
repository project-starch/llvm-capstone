/* Case 4: c81adad105 -- the brigade is not cleaned before the backend
 * connection goes back to the connection pool
 *
 * Shape: holder survives into the next user of a recycled allocator
 * Consumer: modules/proxy/mod_proxy_http.c
 */
#include "../shared/corpus.h"

APRB_CASE(4) {
  apr_pool_t *conn_pool = NULL;
  CHECK(apr_pool_create(&conn_pool, root) == APR_SUCCESS, 750);
  apr_bucket_alloc_t *conn = apr_bucket_alloc_create(conn_pool);
  CHECK(conn, 751);

  struct brigade bb = {.pool = conn_pool};
  void *data = apr_bucket_alloc(64, conn);
  CHECK(data, 752);
  memset(data, 0xA1, 64);
  bb.bucket[bb.n] = data; bb.from[bb.n] = conn; bb.n++;

  /* The fix cleans the brigade before handing the connection back. */
  if (fixed) {
    apr_bucket_free(bb.bucket[0]);
    bb.n = 0;
  }

  unsigned long before = freed_to_malloc;
  /* The connection is handed back and the next request takes it. */
  void *next_user = apr_bucket_alloc(64, conn);
  CHECK(next_user, 753);
  memset(next_user, 0xB2, 64);
  int still_held = bb.n > 0;
  int same = still_held && next_user == bb.bucket[0];
  unsigned char now = ((unsigned char *)data)[0];
  unsigned long freed = freed_to_malloc - before;

  printf("cleaned_before_handback=%d still_held=%d reissued_same_address=%d "
         "now=0x%02X freed_to_malloc=%lu\n", fixed, still_held, same, now, freed);
  APRB_VERDICT(!fixed && still_held && !same && now == 0xA1,
               fixed && !still_held,
               "the brigade still names connection storage after the handback",
               "the brigade was cleaned, so the connection carries nothing over");
  if (still_held) apr_bucket_free(bb.bucket[0]);
  apr_bucket_free(next_user);
  apr_bucket_alloc_destroy(conn);
  return !fixed ? !still_held : !!still_held;
}
