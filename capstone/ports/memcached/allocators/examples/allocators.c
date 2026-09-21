/* One direct-link client, and the smallest program that exercises both seams:
 * an item comes out of a slab chunk, goes back on its class's free list, and
 * the next slabs_alloc pops the same chunk; a read buffer comes out of a
 * cache, goes back on its STAILQ, and the next cache_alloc pops it. Prints
 * what the adapter counted. */
#include "mc_slabs_shim.h"
#include "port.h"
#include "slabs.h"
#include "cache.h"
#include <stdio.h>
#include <stdlib.h>
_Noreturn void mcp_fail(unsigned code) {
  fprintf(stderr, "example failed=%u\n", code);
  exit(1);
}
int main(void) {
  void *metadata = aligned_alloc(4096, MCP_META_BYTES);
  void *payload = aligned_alloc(4096, MCP_PAYLOAD_BYTES);
  if (!metadata || !payload)
    return 1;
  mcp_meta_init(metadata);
  mcp_payload_init(payload);
  mcp_set_mode(0);
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);

  unsigned id = slabs_clsid(sizeof(item) + 100);
  if (!id)
    return 1;
  item *first = slabs_alloc(id, 0);
  if (!first)
    return 1;
  first->nbytes = 100;
  slabs_free(first, id);
  item *second = slabs_alloc(id, 0);
  if (!second)
    return 1;
  int same_chunk = first == second;
  slabs_free(second, id);

  cache_t *rbufs = cache_create("rbuf", 16384, sizeof(char *));
  if (!rbufs)
    return 1;
  char *buffer = cache_alloc(rbufs);
  if (!buffer)
    return 1;
  buffer[0] = 'x';
  cache_free(rbufs, buffer);
  char *again = cache_alloc(rbufs);
  int same_object = buffer == again;
  cache_free(rbufs, again);
  cache_destroy(rbufs);

  struct mcp_header h = {0};
  mcp_stats(&h);
  printf("MCP example same_chunk=%d same_object=%d pages=%llu chunk_reuses=%llu "
         "chunk_releases=%llu object_reuses=%llu object_releases=%llu\n",
         same_chunk, same_object, (unsigned long long)h.pages,
         (unsigned long long)h.chunk_reuses, (unsigned long long)h.chunk_releases,
         (unsigned long long)h.object_reuses, (unsigned long long)h.object_releases);
  /* The reissues are the allocators' own behaviour and the reason the corpus
   * exists; an example that did not see them would be measuring the wrong
   * allocators. */
  if (!(same_chunk && same_object && h.chunk_reuses >= 1 && h.object_reuses >= 1))
    return 1;
  printf("ALLOCATOR_EXAMPLE memcached PASS pointer_bytes=%zu\n", sizeof(void *));
  return 0;
}
