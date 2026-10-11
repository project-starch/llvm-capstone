/* A direct-link client of the bucket allocator, natively. It must observe the
 * reuse the corpus exists for: a freed small node comes back first, at the
 * same address, from the allocator's own LIFO -- and the adapter's counters
 * say so. Without that the example would be measuring the wrong allocator. */
#include "port.h"
#include "apr_shim.h"
#include "apr_pools.h"
#include "apr_bucket_shim.h"
#include <stdio.h>
#include <stdlib.h>

_Noreturn void aprp_fail(unsigned code) {
  fprintf(stderr, "APRP failed=%u\n", code);
  exit(1);
}
void aprp_replay(const struct aprp_header *input, struct aprp_header *out) {
  (void)input;
  (void)out;
}
int main(void) {
  void *metadata = aligned_alloc(4096, APRP_META_BYTES);
  void *payload = aligned_alloc(4096, APRP_PAYLOAD_BYTES);
  if (!metadata || !payload)
    return 2;
  aprp_meta_init(metadata);
  aprp_payload_init(payload);
  aprp_set_mode(0);
  if (apr_pool_initialize() != APR_SUCCESS)
    return 2;
  apr_pool_t *pool = NULL;
  if (apr_pool_create(&pool, NULL) != APR_SUCCESS)
    return 2;
  apr_bucket_alloc_t *ba = apr_bucket_alloc_create(pool);
  unsigned char *a = apr_bucket_alloc(64, ba), *b = apr_bucket_alloc(64, ba);
  if (!ba || !a || !b || a == b)
    return 3;
  apr_bucket_free(a);
  apr_bucket_free(b);
  unsigned char *c = apr_bucket_alloc(64, ba); /* LIFO: b comes back first */
  unsigned char *d = apr_bucket_alloc(64, ba);
  unsigned long pieces, reissues, files;
  aprb_stats(&pieces, &reissues, &files);
  int lifo = c == b && d == a;
  /* A large node is a whole pool node and never touches the freelist. */
  unsigned char *big = apr_bucket_alloc(4096, ba);
  apr_bucket_free(big);
  struct aprp_header stats = {0};
  aprp_stats(&stats);
  printf("lifo_reissue=%d pieces=%lu files=%lu reissues=%lu node_releases=%llu\n", lifo,
         pieces, files, reissues, (unsigned long long)stats.node_releases);
  apr_bucket_alloc_destroy(ba);
  apr_pool_destroy(pool);
  apr_pool_terminate();
  return !(lifo && reissues == 2 && files == 2 && stats.node_releases >= 1);
}
