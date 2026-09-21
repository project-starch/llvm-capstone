/* Case 5: 60919177e8 -- "a read from a deleted brigade ... an ap_get_brigade()
 * call after the brigade had been destroyed"
 *
 * Shape: the holder itself is used after its pool went
 * Consumer: server/protocol.c, ap_rgetline()'s folding path
 */
#include "../shared/corpus.h"

APRB_CASE(5) {
  apr_pool_t *bb_pool = NULL;
  CHECK(apr_pool_create(&bb_pool, root) == APR_SUCCESS, 760);
  struct brigade *bb = apr_palloc(bb_pool, sizeof(*bb));
  CHECK(bb, 761);
  memset(bb, 0, sizeof(*bb));
  bb->pool = bb_pool;
  bb->n = 3;

  unsigned long before = freed_to_malloc;
  /* The folding path destroyed the brigade and then read from it again. The
   * fix stops using it after the destroy. */
  apr_pool_destroy(bb_pool);
  struct brigade *reader = fixed ? NULL : bb;

  apr_pool_t *other = NULL;
  CHECK(apr_pool_create(&other, root) == APR_SUCCESS, 762);
  struct brigade *taken = apr_palloc(other, sizeof(*taken));
  CHECK(taken, 763);
  memset(taken, 0x5A, sizeof(*taken));      /* the next holder writes its own */
  int reissued = (void *)taken == (void *)bb;
  int read_after_destroy = reader != NULL;
  unsigned long freed = freed_to_malloc - before;

  printf("read_after_destroy=%d holder_reissued=%d freed_to_malloc=%lu\n",
         read_after_destroy, reissued, freed);
  APRB_VERDICT(!fixed && read_after_destroy && reissued, fixed && !read_after_destroy,
               "the destroyed brigade's storage already belongs to another holder",
               "nothing reads the brigade after it is destroyed");
  return !fixed ? !(read_after_destroy && reissued) : !!read_after_destroy;
}
