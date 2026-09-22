/* Case 6: 4930450013 -- "Fix bucket lifetimes so that they don't live longer
 * than their brigades"
 *
 * Shape: the contents outlive the holder, and the allocator goes with it
 * Consumer: modules/http/http_filters.c, the chunk filter
 */
#include "../shared/corpus.h"

APRB_CASE(6) {
  o->defect_text = "a bucket is still named after its brigade and allocator went";
  o->fixed_text = "the bucket was released while its brigade was alive";
  apr_pool_t *bb_pool = NULL;
  CHECK(apr_pool_create(&bb_pool, root) == APR_SUCCESS, 770);
  apr_bucket_alloc_t *ba = apr_bucket_alloc_create(bb_pool);
  CHECK(ba, 771);
  unsigned char *data = apr_bucket_alloc(64, ba);
  CHECK(data, 772);
  memset(data, 0xA1, 64);

  /* The filter kept the bucket past the brigade it belonged to. The fix
   * releases it while the brigade, and therefore the allocator, is alive. */
  unsigned char *kept = data;
  if (fixed) {
    apr_bucket_free(data);
    kept = NULL;
  }

  apr_pool_destroy(bb_pool); /* the brigade goes, and its allocator with it */
  apr_pool_t *other = NULL;
  CHECK(apr_pool_create(&other, root) == APR_SUCCESS, 773);
  unsigned char *reissued = apr_palloc(other, 64);
  CHECK(reissued, 774);
  memset(reissued, 0xB2, 64);
  o->still_held = kept != NULL;
  if (o->still_held) {
    held = kept;
    mark(6);
    o->now = read_probe(held);
  }
  o->defect = o->still_held;
  o->held_up = !o->still_held;
}
