/* One direct-link client, and the smallest program that exercises the seam:
 * a pool comes up out of a node, is destroyed back onto the free list, and
 * the next pool pops the same node. Prints what the adapter counted. */
#include "port.h"
#include "apr_shim.h"
#include "apr_pools.h"
#include <stdio.h>
#include <stdlib.h>
_Noreturn void aprp_fail(unsigned code) {
  fprintf(stderr, "example failed=%u\n", code);
  exit(1);
}
int main(void) {
  void *metadata = aligned_alloc(4096, APRP_META_BYTES);
  void *payload = aligned_alloc(4096, APRP_PAYLOAD_BYTES);
  if (!metadata || !payload)
    return 1;
  aprp_meta_init(metadata);
  aprp_payload_init(payload);
  aprp_set_mode(0);
  if (apr_pool_initialize() != APR_SUCCESS)
    return 1;
  apr_pool_t *first = NULL, *second = NULL;
  if (apr_pool_create(&first, NULL) != APR_SUCCESS)
    return 1;
  char *p = apr_palloc(first, 100);
  if (!p)
    return 1;
  p[0] = 'x';
  apr_pool_destroy(first);
  if (apr_pool_create(&second, NULL) != APR_SUCCESS)
    return 1;
  int same_node = (void *)first == (void *)second;
  apr_pool_destroy(second);
  apr_pool_terminate();
  struct aprp_header h = {0};
  aprp_stats(&h);
  printf("APRP example same_node=%d nodes=%llu reuses=%llu releases=%llu "
         "discards=%llu\n",
         same_node, (unsigned long long)h.nodes,
         (unsigned long long)h.node_reuses,
         (unsigned long long)h.node_releases,
         (unsigned long long)h.node_discards);
  /* The reissue is the allocator's own behaviour and the reason the corpus
   * exists; an example that did not see it would be measuring the wrong
   * allocator. */
  return same_node && h.node_reuses >= 1 ? 0 : 1;
}
