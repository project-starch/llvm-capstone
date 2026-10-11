/* Case 2: 106d0761c0 -- a brigade on r->pool has to carry EOR past the request
 *
 * Shape: holder on a pool that dies before its contents are needed
 * Consumer: server/protocol.c, ap_request_core_filter()
 */
#include "../shared/corpus.h"

APRB_CASE(2) {
  o->defect_text = "the brigade holding EOR was reissued to the next request";
  o->fixed_text = "the holder is on the connection pool and survives the request";
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
  uintptr_t holder = (uintptr_t)bb;

  apr_pool_destroy(req_pool);  /* the request ends */
  apr_pool_t *next_req = NULL; /* the next one on the same connection */
  CHECK(apr_pool_create(&next_req, conn_pool) == APR_SUCCESS, 734);
  unsigned char *reissued = apr_palloc(next_req, sizeof(*bb));
  CHECK(reissued, 735);
  memset(reissued, 0x5A, sizeof(*bb));
  o->reissued_same_address = (uintptr_t)reissued == holder;
  if (!fixed) {
    /* The core output filter reads the brigade it was handed for EOR after
     * the request pool that held it is gone. */
    CHECK(o->reissued_same_address, 736);
    held = (volatile unsigned char *)bb;
    mark(2);
    o->now = read_probe(held);
    o->read_after_destroy = 1;
  }
  o->defect = o->read_after_destroy && o->reissued_same_address;
  o->held_up = !o->reissued_same_address;
}
