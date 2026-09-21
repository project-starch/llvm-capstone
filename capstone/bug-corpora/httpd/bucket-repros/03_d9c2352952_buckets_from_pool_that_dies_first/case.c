/* Case 3: d9c2352952 -- "buckets inserted to it can be created from scpool and
 * this pool can be freed before this brigade"
 *
 * Shape: payload from a pool that dies before its holder
 * Consumer: modules/ssl, the brigade outliving scpool
 */
#include "../shared/corpus.h"

APRB_CASE(3) {
  o->defect_text = "the brigade still names storage the destroyed pool gave back";
  o->fixed_text = "the brigade was cleaned before the pool went";
  apr_pool_t *long_pool = NULL, *scpool = NULL;
  CHECK(apr_pool_create(&long_pool, root) == APR_SUCCESS, 740);
  CHECK(apr_pool_create(&scpool, root) == APR_SUCCESS, 741);

  struct brigade *bb = apr_palloc(long_pool, sizeof(*bb));
  CHECK(bb, 742);
  memset(bb, 0, sizeof(*bb));
  bb->pool = long_pool;
  unsigned char *payload = apr_palloc(scpool, 64); /* created from scpool */
  CHECK(payload, 743);
  memset(payload, 0xA1, 64);
  bb->bucket[bb->n++] = payload;

  /* The fix cleans the brigade before scpool goes; the defect leaves it. */
  if (fixed)
    bb->n = 0;

  apr_pool_destroy(scpool);
  apr_pool_t *other = NULL;
  CHECK(apr_pool_create(&other, root) == APR_SUCCESS, 744);
  unsigned char *reissued = apr_palloc(other, 64);
  CHECK(reissued, 745);
  memset(reissued, 0xB2, 64);
  o->still_held = bb->n > 0;
  o->reissued_same_address = reissued == payload;
  if (o->still_held) {
    CHECK(o->reissued_same_address, 746);
    held = payload;
    mark(3);
    o->now = read_probe(held);
  }
  o->defect = o->still_held && o->reissued_same_address && o->now == 0xB2;
  o->held_up = !o->still_held;
}
