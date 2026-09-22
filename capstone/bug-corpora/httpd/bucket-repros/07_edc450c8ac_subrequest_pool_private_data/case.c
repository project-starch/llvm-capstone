/* Case 7: edc450c8ac -- "buckets associated with a subrequest having private
 * data in the wrong (i.e., subrequest) pool, leading to a segfault later in
 * processing the main request"
 *
 * Shape: private data on a pool shorter-lived than the bucket holding it
 * Consumer: server/request.c and the subrequest path
 */
#include "../shared/corpus.h"

APRB_CASE(7) {
  o->defect_text = "the main request reads private data the subrequest gave back";
  o->fixed_text = "the private data is on the main request's pool and survives";
  apr_pool_t *main_pool = NULL, *sub_pool = NULL;
  CHECK(apr_pool_create(&main_pool, root) == APR_SUCCESS, 780);
  CHECK(apr_pool_create(&sub_pool, root) == APR_SUCCESS, 781);

  /* The committed fix allocates the private data from the MAIN request's pool;
   * the defect used the subrequest's. */
  apr_pool_t *priv_pool = fixed ? main_pool : sub_pool;
  unsigned char *priv = apr_palloc(priv_pool, 64);
  CHECK(priv, 782);
  memset(priv, 0xA1, 64);

  struct brigade *bb = apr_palloc(main_pool, sizeof(*bb)); /* held by the main request */
  CHECK(bb, 783);
  memset(bb, 0, sizeof(*bb));
  bb->pool = main_pool;
  bb->bucket[bb->n++] = priv;

  apr_pool_destroy(sub_pool); /* the subrequest finishes */
  apr_pool_t *later = NULL;
  CHECK(apr_pool_create(&later, root) == APR_SUCCESS, 784);
  unsigned char *reissued = apr_palloc(later, 64);
  CHECK(reissued, 785);
  memset(reissued, 0xB2, 64);
  o->reissued_same_address = reissued == priv;
  if (!fixed) {
    CHECK(o->reissued_same_address, 786);
    held = bb->bucket[0];
    mark(7);
    o->now = read_probe(held);
  }
  o->defect = o->reissued_same_address && o->now == 0xB2;
  o->held_up = !o->reissued_same_address;
}
