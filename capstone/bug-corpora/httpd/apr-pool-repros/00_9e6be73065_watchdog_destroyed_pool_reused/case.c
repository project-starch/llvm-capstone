/* Case 0: 9e6be73065 -- mod_watchdog reuses a pool it destroyed
 *
 * Shape: stale allocator handle / reuse / allocation through the dead handle
 * Consumer: modules/core/mod_watchdog.c, wd_worker()
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

APR_CASE(0) {
  o->defect_text = "a destroyed pool's handle now names a live pool";
  o->fixed_text = "the handle was cleared, the loop takes a fresh pool";

  apr_pool_t *ctx = NULL;
  CHECK(apr_pool_create(&ctx, root) == APR_SUCCESS, 710);
  char *mine = apr_palloc(ctx, 64);
  CHECK(mine, 711);
  memset(mine, 0xA1, 64);

  /* The handle the worker loop keeps across the destroy. Its address, not the
   * pointer: in the protected arm the pointer itself is dead afterwards. */
  uintptr_t destroyed = (uintptr_t)ctx;
  held = (volatile unsigned char *)ctx;
  apr_pool_destroy(ctx);
  if (fixed)
    ctx = NULL; /* upstream 9e6be73065 */

  /* the next iteration of the worker loop creates its own pool */
  apr_pool_t *other = NULL;
  CHECK(apr_pool_create(&other, root) == APR_SUCCESS, 712);
  char *theirs = apr_palloc(other, 64);
  CHECK(theirs, 713);
  memset(theirs, 0xB2, 64);
  o->pool_struct_reissued = destroyed == (uintptr_t)other;

  if (ctx) {
    /* The loop's `if (!ctx)` sees a live-looking handle, and the first thing
     * it does with it is read the pool struct. That read is the probe. The
     * setup that makes it a stale access -- the same node came back as
     * `other` -- is checked BEFORE the marker, so the marker's presence is
     * itself evidence that reuse happened. */
    CHECK(o->pool_struct_reissued, 714);
    mark(0);
    (void)read_probe(held);
    char *stale = apr_palloc(ctx, 64);
    if (stale) {
      memset(stale, 0xCC, 64);
      o->allocated_through_stale = 1;
    }
  }
  for (int i = 0; i < 64; i++)
    if ((unsigned char)theirs[i] != 0xB2) {
      o->other_pool_corrupted = 1;
      break;
    }
}
