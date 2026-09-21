/* Case 3: d9c2352952 -- "buckets inserted to it can be created from scpool and
 * this pool can be freed before this brigade"
 *
 * Shape: payload from a pool that dies before its holder
 * Consumer: modules/ssl, the brigade outliving scpool
 */
#include "../shared/corpus.h"

APRB_CASE(3) {
  apr_pool_t *long_pool = NULL, *scpool = NULL;
  CHECK(apr_pool_create(&long_pool, root) == APR_SUCCESS, 740);
  CHECK(apr_pool_create(&scpool, root) == APR_SUCCESS, 741);

  struct brigade *bb = apr_palloc(long_pool, sizeof(*bb));
  CHECK(bb, 742);
  memset(bb, 0, sizeof(*bb));
  bb->pool = long_pool;
  void *payload = apr_palloc(scpool, 64);   /* created from scpool */
  CHECK(payload, 743);
  memset(payload, 0xA1, 64);
  bb->bucket[bb->n++] = payload;

  /* The fix cleans the brigade before scpool goes; the defect leaves it. */
  if (fixed)
    bb->n = 0;

  unsigned long before = freed_to_malloc;
  apr_pool_destroy(scpool);
  apr_pool_t *other = NULL;
  CHECK(apr_pool_create(&other, root) == APR_SUCCESS, 744);
  void *reissued = apr_palloc(other, 64);
  CHECK(reissued, 745);
  memset(reissued, 0xB2, 64);
  int still_held = bb->n > 0;
  int same = reissued == payload;
  unsigned char now = ((unsigned char *)payload)[0];
  unsigned long freed = freed_to_malloc - before;

  printf("brigade_still_holds=%d reissued_same_address=%d now=0x%02X freed_to_malloc=%lu\n",
         still_held, same, now, freed);
  APRB_VERDICT(!fixed && still_held && same && now == 0xB2, fixed && !still_held,
               "the brigade still names storage the destroyed pool gave back",
               "the brigade was cleaned before the pool went");
  return !fixed ? !(still_held && same && now == 0xB2) : !!still_held;
}
