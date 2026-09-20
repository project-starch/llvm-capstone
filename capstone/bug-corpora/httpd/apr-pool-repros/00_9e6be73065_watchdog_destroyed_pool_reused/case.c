/* Case 0: 9e6be73065 -- mod_watchdog reuses a pool it destroyed
 *
 * Shape: stale allocator handle / reuse / allocation through the dead handle
 * Consumer: modules/core/mod_watchdog.c, wd_worker()
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

APR_CASE(0) {
  apr_pool_t *ctx = NULL;
  CHECK(apr_pool_create(&ctx, root) == APR_SUCCESS, 710);
  char *mine = apr_palloc(ctx, 64);
  CHECK(mine, 711);
  memset(mine, 0xA1, 64);

  apr_pool_t *destroyed = ctx;
  apr_pool_destroy(ctx);
  if (fixed)
    ctx = NULL; /* upstream 9e6be73065 */

  /* the next iteration of the worker loop creates its own pool */
  apr_pool_t *other = NULL;
  CHECK(apr_pool_create(&other, root) == APR_SUCCESS, 712);
  char *theirs = apr_palloc(other, 64);
  CHECK(theirs, 713);
  memset(theirs, 0xB2, 64);

  int same_struct = (void *)destroyed == (void *)other;
  int reused = 0;
  if (ctx) { /* the loop's `if (!ctx)` sees a live-looking handle */
    char *stale = apr_palloc(ctx, 64);
    if (stale) {
      memset(stale, 0xCC, 64);
      reused = 1;
    }
  }
  int corrupted = 0;
  for (int i = 0; i < 64; i++)
    if ((unsigned char)theirs[i] != 0xB2) {
      corrupted = 1;
      break;
    }

  printf("pool_struct_reissued=%d allocated_through_stale=%d "
         "other_pool_corrupted=%d freed_to_malloc=0\n",
         same_struct, reused, corrupted);
  APR_VERDICT(!fixed && same_struct && reused, fixed && !reused,
              "a destroyed pool's handle now names a live pool",
              "the handle was cleared, the loop takes a fresh pool");
  return !fixed ? !(same_struct && reused) : !!reused;
}
