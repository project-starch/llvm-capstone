/* Case 2: 106d0761c0 -- a brigade on r->pool has to carry EOR past the request
 *
 * Shape: holder on a pool that dies before its contents are needed
 * Consumer: server/protocol.c, ap_request_core_filter()
 */
#include "../shared/corpus.h"

APRB_CASE(2) {
  apr_pool_t *conn_pool = NULL, *req_pool = NULL;
  CHECK(apr_pool_create(&conn_pool, root) == APR_SUCCESS, 730);
  CHECK(apr_pool_create(&req_pool, conn_pool) == APR_SUCCESS, 731);

  /* "For EOR it can't use a brigade created on r->pool": the holder must
   * outlive the request, so the fix puts it on c->pool. */
  apr_pool_t *holder_pool = fixed ? conn_pool : req_pool;
  struct brigade *bb = apr_palloc(holder_pool, sizeof(*bb));
  CHECK(bb, 732);
  memset(bb, 0, sizeof(*bb));
  bb->pool = holder_pool;
  bb->bucket[bb->n++] = apr_palloc(conn_pool, 64); /* the EOR payload itself */
  CHECK(bb->bucket[0], 733);
  memset(bb->bucket[0], 0xA1, 64);

  unsigned long before = freed_to_malloc;
  apr_pool_destroy(req_pool);              /* the request ends */
  apr_pool_t *next_req = NULL;             /* the next one on the same connection */
  CHECK(apr_pool_create(&next_req, conn_pool) == APR_SUCCESS, 734);
  void *reissued = apr_palloc(next_req, sizeof(*bb));
  CHECK(reissued, 735);
  int holder_reissued = reissued == (void *)bb;
  unsigned long freed = freed_to_malloc - before;

  printf("holder_pool=%s holder_reissued=%d freed_to_malloc=%lu\n",
         fixed ? "connection" : "request", holder_reissued, freed);
  APRB_VERDICT(!fixed && holder_reissued, fixed && !holder_reissued,
               "the brigade holding EOR was reissued to the next request",
               "the holder is on the connection pool and survives the request");
  return !fixed ? !holder_reissued : !!holder_reissued;
}
