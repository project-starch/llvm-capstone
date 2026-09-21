/* Case 5: 60919177e8 -- "a read from a deleted brigade ... an ap_get_brigade()
 * call after the brigade had been destroyed"
 *
 * Shape: the holder itself is used after its pool went
 * Consumer: server/protocol.c, ap_rgetline()'s folding path
 */
#include "../shared/corpus.h"

APRB_CASE(5) {
  o->defect_text = "the destroyed brigade's storage already belongs to another holder";
  o->fixed_text = "nothing reads the brigade after it is destroyed";
  apr_pool_t *bb_pool = NULL;
  CHECK(apr_pool_create(&bb_pool, root) == APR_SUCCESS, 760);
  struct brigade *bb = apr_palloc(bb_pool, sizeof(*bb));
  CHECK(bb, 761);
  memset(bb, 0, sizeof(*bb));
  bb->pool = bb_pool;
  bb->n = 3;
  uintptr_t holder = (uintptr_t)bb;

  /* The folding path destroyed the brigade and then read from it again. The
   * fix stops using it after the destroy. */
  apr_pool_destroy(bb_pool);
  apr_pool_t *other = NULL;
  CHECK(apr_pool_create(&other, root) == APR_SUCCESS, 762);
  unsigned char *taken = apr_palloc(other, sizeof(*bb));
  CHECK(taken, 763);
  memset(taken, 0x5A, sizeof(*bb)); /* the next holder writes its own */
  o->reissued_same_address = (uintptr_t)taken == holder;
  if (!fixed) {
    CHECK(o->reissued_same_address, 764);
    held = (volatile unsigned char *)bb;
    mark(5);
    o->now = read_probe(held);
    o->read_after_destroy = 1;
  }
  o->defect = o->read_after_destroy && o->reissued_same_address;
  o->held_up = !o->read_after_destroy;
}
