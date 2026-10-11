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

  /* THE HANDBACK IS THE LENDER'S EPOCH. Upstream's connection_cleanup puts the
   * backend connection back on the reslist and leaves its bucket allocator as
   * it is: the rule that nothing of the old request may outlive the handback
   * is a copying discipline (ap_proxy_buckets_lifetime_transform, then
   * cleanup), not an allocator event. This corpus reduces the consumer to
   * the lender's contract, and under the Sublet discipline a lender that
   * reuses without freeing revokes at the point of reuse. So the reduced
   * consumer ends the allocator's tenancy here -- the operation the lender
   * would perform -- and the next request takes the connection under a
   * fresh one. That is the one step upstream does not make; PROVENANCE.md
   * says so. Both arms take it: the fix differential is the cleanup above. */
  apr_bucket_alloc_destroy(conn);
  conn = apr_bucket_alloc_create(conn_pool);
  CHECK(conn, 753);

  /* The next request takes the connection. */
  unsigned char *next_user = apr_bucket_alloc(64, conn);
  CHECK(next_user, 754);
  memset(next_user, 0xB2, 64);
  o->still_held = bb.n > 0;
  o->reissued_same_address = o->still_held && next_user == data;
  if (o->still_held) {
    /* The old brigade is touched again: under Sublet the epoch has ended
     * the bucket's alias, and this read is where the class-3 defect is
     * caught -- no free happened, so no free-triggered mechanism sees it. */
    held = data;
    mark(4);
    o->now = read_probe(held);
  }
  apr_bucket_free(next_user);
  apr_bucket_alloc_destroy(conn);
  o->defect = o->still_held && o->reissued_same_address;
  o->held_up = !o->still_held;
}
