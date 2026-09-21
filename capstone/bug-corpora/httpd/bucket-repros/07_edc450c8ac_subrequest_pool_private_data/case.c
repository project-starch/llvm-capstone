/* Case 7: edc450c8ac -- "buckets associated with a subrequest having private
 * data in the wrong (i.e., subrequest) pool, leading to a segfault later in
 * processing the main request"
 *
 * Shape: private data on a pool shorter-lived than the bucket holding it
 * Consumer: server/request.c and the subrequest path
 */
#include "../shared/corpus.h"

APRB_CASE(7) {
  apr_pool_t *main_pool = NULL, *sub_pool = NULL;
  CHECK(apr_pool_create(&main_pool, root) == APR_SUCCESS, 780);
  CHECK(apr_pool_create(&sub_pool, root) == APR_SUCCESS, 781);

  /* The committed fix allocates the private data from the MAIN request's pool;
   * the defect used the subrequest's. */
  apr_pool_t *priv_pool = fixed ? main_pool : sub_pool;
  void *priv = apr_palloc(priv_pool, 64);
  CHECK(priv, 782);
  memset(priv, 0xA1, 64);

  struct brigade *bb = apr_palloc(main_pool, sizeof(*bb));   /* held by the main request */
  CHECK(bb, 783);
  memset(bb, 0, sizeof(*bb));
  bb->pool = main_pool;
  bb->bucket[bb->n++] = priv;

  unsigned long before = freed_to_malloc;
  apr_pool_destroy(sub_pool);          /* the subrequest finishes */
  apr_pool_t *later = NULL;
  CHECK(apr_pool_create(&later, root) == APR_SUCCESS, 784);
  void *reissued = apr_palloc(later, 64);
  CHECK(reissued, 785);
  memset(reissued, 0xB2, 64);
  int same = reissued == priv;
  unsigned char now = ((unsigned char *)bb->bucket[0])[0];
  unsigned long freed = freed_to_malloc - before;

  printf("private_data_pool=%s reissued_same_address=%d now=0x%02X freed_to_malloc=%lu\n",
         fixed ? "main" : "subrequest", same, now, freed);
  APRB_VERDICT(!fixed && same && now == 0xB2, fixed && !same,
               "the main request reads private data the subrequest gave back",
               "the private data is on the main request's pool and survives");
  return !fixed ? !(same && now == 0xB2) : !!same;
}
