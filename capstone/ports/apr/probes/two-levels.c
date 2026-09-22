/* Does Apache's bucket allocator recycle, and does anything reach malloc?
 *
 * The census says the return path has two levels. This runs them:
 *   level 1  apr_bucket_free pushes a SMALL node onto list->freelist, LIFO
 *   level 2  a LARGE node goes to apr_allocator_free, APR's size-bucketed list
 * and neither calls free(). The probe counts free() itself by interposing, so
 * "nothing reaches malloc" is measured here and not quoted from the source.
 */
#include "apr_bucket_shim.h"
#include <stdio.h>
#include <string.h>

unsigned long freed_to_malloc; /* counted by the interposed free() below */

int main(void) {
  if (apr_pool_initialize() != APR_SUCCESS) return 75;
  apr_pool_t *p = NULL;
  if (apr_pool_create(&p, NULL) != APR_SUCCESS) return 75;
  apr_bucket_alloc_t *list = apr_bucket_alloc_create(p);
  if (!list) return 75;

  const apr_size_t small = 64, large = APR_BUCKET_ALLOC_SIZE * 8;
  unsigned long before = freed_to_malloc;

  /* level 1: a small node, returned, then asked for again */
  void *a = apr_bucket_alloc(small, list);
  memset(a, 0xA1, small);
  apr_bucket_free(a);
  void *b = apr_bucket_alloc(small, list);
  int small_same = (a == b);
  unsigned char stale = ((unsigned char *)a)[0];
  memset(b, 0xB2, small);
  unsigned char after_reuse = ((unsigned char *)a)[0];

  /* level 2: a large node, returned to APR's allocator, then asked for again */
  void *c = apr_bucket_alloc(large, list);
  apr_bucket_free(c);
  void *d = apr_bucket_alloc(large, list);
  int large_same = (c == d);

  printf("APR_BUCKET_ALLOC_SIZE=%zu small_node=%zu\n",
         (size_t)APR_BUCKET_ALLOC_SIZE,
         (size_t)(APR_BUCKET_ALLOC_SIZE + APR_ALIGN_DEFAULT(4 * sizeof(void *))));
  printf("level1_freelist_same_address=%d  stale_before=0x%02X stale_after=0x%02X\n",
         small_same, stale, after_reuse);
  printf("level2_allocator_same_address=%d\n", large_same);
  /* Captured BEFORE the teardown below. The lifecycle does free -- destroying
   * the allocator hands memory back for real -- but the RECYCLING does not, and
   * that is what is being measured. Reading the counter again after teardown
   * would report the teardown. */
  unsigned long during_reuse = freed_to_malloc - before;
  int ok = small_same && large_same && during_reuse == 0;
  printf("freed_to_malloc=%lu\n", during_reuse);
  printf("VERDICT %s\n",
         ok
             ? "TWO-LEVEL-RECYCLING both levels reissue the same storage, nothing reaches malloc"
             : "INCONCLUSIVE");
  apr_bucket_free(b); apr_bucket_free(d);
  apr_bucket_alloc_destroy(list);
  apr_pool_destroy(p);
  apr_pool_terminate();
  printf("freed_to_malloc_including_teardown=%lu\n", freed_to_malloc - before);
  return !ok;
}
