/* mod_watchdog's destroyed-pool reuse (upstream 9e6be73065), reduced to its
 * allocator call sequence against the real apr_pools.c.
 *
 * wd_worker() created a pool per iteration, destroyed it, and did not clear the
 * variable; a later `if (!ctx)` therefore saw a live-looking handle and reused
 * a destroyed pool. APR does not return a destroyed pool's nodes to malloc:
 * allocator_free (apr_pools.c:414) pushes them onto a size-bucketed LIFO free
 * list, and the next apr_pool_create pops the same node. So the stale handle
 * does not merely dangle -- it comes to name whichever pool got that node.
 */
#include "apr_pools.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (apr_pool_initialize() != APR_SUCCESS)
    return 75;
  apr_pool_t *root = NULL;
  if (apr_pool_create(&root, NULL) != APR_SUCCESS)
    return 75;

  apr_pool_t *ctx = NULL;
  if (apr_pool_create(&ctx, root) != APR_SUCCESS)
    return 75;
  char *mine = apr_palloc(ctx, 64);
  if (!mine) return 75;
  memset(mine, 0xA1, 64);
  apr_pool_t *destroyed = ctx;
  apr_pool_destroy(ctx);
  if (fixed)
    ctx = NULL; /* upstream 9e6be73065 */

  /* the next iteration of the worker loop creates its own pool */
  apr_pool_t *other = NULL;
  if (apr_pool_create(&other, root) != APR_SUCCESS)
    return 75;
  char *theirs = apr_palloc(other, 64);
  if (!theirs) return 75;
  memset(theirs, 0xB2, 64);

  int same_struct = (void *)destroyed == (void *)other;
  int reused = 0;
  char *stale = NULL;
  if (ctx) { /* the loop's `if (!ctx)` sees a live-looking handle */
    stale = apr_palloc(ctx, 64);
    if (stale) { memset(stale, 0xCC, 64); reused = 1; }
  }
  int corrupted = 0;
  for (int i = 0; i < 64; i++)
    if ((unsigned char)theirs[i] != 0xB2) { corrupted = 1; break; }

  printf("arm=%s pool_struct_reissued=%d allocated_through_stale=%d "
         "other_pool_corrupted=%d freed_to_malloc=0\n",
         fixed ? "fixed" : "buggy", same_struct, reused, corrupted);
  printf("VERDICT %s\n",
         !fixed && same_struct && reused
             ? "DEFECT-REPRODUCED a destroyed pool's handle now names a live pool"
         : fixed && !reused ? "FIXED the handle was cleared, the loop takes a fresh pool"
                            : "INCONCLUSIVE");
  apr_pool_destroy(root);
  apr_pool_terminate();
  return !fixed ? !(same_struct && reused) : !!reused;
}
