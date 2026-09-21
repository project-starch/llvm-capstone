/* Case 4: c81adad105 -- the brigade is not cleaned before the backend
 * connection goes back to the connection pool
 *
 * Shape: holder survives into the next user of a recycled allocator
 * Consumer: modules/proxy/mod_proxy_http.c
 */
#include "../shared/corpus.h"

APRB_CASE(4) {
  o->defect_text = "the brigade still names connection storage after the handback";
  o->fixed_text = "the brigade was cleaned, so the connection carries nothing over";
  apr_pool_t *conn_pool = NULL;
  CHECK(apr_pool_create(&conn_pool, root) == APR_SUCCESS, 750);
  apr_bucket_alloc_t *conn = apr_bucket_alloc_create(conn_pool);
  CHECK(conn, 751);

  struct brigade bb = {.pool = conn_pool};
  unsigned char *data = apr_bucket_alloc(64, conn);
  CHECK(data, 752);
  memset(data, 0xA1, 64);
  bb.bucket[bb.n] = data;
  bb.from[bb.n] = conn;
  bb.n++;

  /* The fix cleans the brigade before handing the connection back. */
  if (fixed) {
    apr_bucket_free(bb.bucket[0]);
    bb.n = 0;
  }

  /* The connection is handed back and the next request takes it. In the
   * reduced sequence nothing ends the old bucket's lifetime: the defect is a
   * bucket carried over live, and the read that follows is of storage still
   * allocated. No arm is expected to fault here, and the case says so. */
  unsigned char *next_user = apr_bucket_alloc(64, conn);
  CHECK(next_user, 753);
  memset(next_user, 0xB2, 64);
  o->still_held = bb.n > 0;
  o->reissued_same_address = o->still_held && next_user == data;
  if (o->still_held) {
    held = data;
    mark(4);
    o->now = read_probe(held);
    apr_bucket_free(bb.bucket[0]);
  }
  apr_bucket_free(next_user);
  apr_bucket_alloc_destroy(conn);
  o->defect = o->still_held && !o->reissued_same_address && o->now == 0xA1;
  o->held_up = !o->still_held;
}
