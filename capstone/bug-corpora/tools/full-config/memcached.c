/* FULL CONFIGURATION, memcached: the Sublet port of slabs.c and cache.c (the allocators port's ledger
 * src/shared/leases.c, metadata heap and Sublet authority, as ports/memcached/app's slab-sublet arm
 * links them) is brought up before main() exactly as mcapp-slab-sublet.c does -- a 64 MiB payload
 * lent LINEAR by the Sublet heap, 16 MiB of metadata from malloc, mode 1 -- and one slab page is
 * carved into chunks, one chunk issued (a renew: revoke and a fresh alias), written and released (a
 * revoke). The ledger stays live for the whole case.
 *
 * The case's objects come straight from malloc, not from a slab, so this measures that the port
 * changes nothing for a direct-allocation bug. */
#include <stdio.h>
#include <stdlib.h>

#include "port.h"

unsigned long __capstone_sublet_malloc_linear(size_t n, capstone_cap_slot *out);

_Noreturn void mcp_fail(unsigned code) {
  printf("FULLCONFIG-FAILED memcached: ledger code %u\n", code);
  fflush(stdout);
  exit(75);
}

__attribute__((constructor)) static void full_config_memcached(void) {
  static capstone_cap_slot payload;
  void *metadata = malloc(MCP_META_BYTES);
  if (!metadata || !__capstone_sublet_malloc_linear(MCP_PAYLOAD_BYTES, &payload))
    mcp_fail(901);
  mcp_meta_init(metadata);
  mcp_payload_init(payload.c); /* handed over once, as mcapp-slab-sublet.c does */
  mcp_set_mode(1);
  void *page = mcp_page_backing((size_t)1 << 20);
  if (!page)
    mcp_fail(902);
  void *first = mcp_page_carve(page, 1, 128, 64);
  mcp_chunk_release(first, 1);          /* filed on the free list, as the split does */
  unsigned char *chunk = mcp_chunk_issue(first); /* a renew: revoke, fresh alias */
  chunk[64] = 1;
  mcp_chunk_release(chunk, 1);          /* a revoke */
  struct mcp_header h;
  mcp_stats(&h);
  printf("FULLCONFIG memcached slabs=sublet-port live mode=1 pages=%llu chunk_releases=%llu\n",
         (unsigned long long)h.pages, (unsigned long long)h.chunk_releases);
  fflush(stdout);
  if (h.pages != 1 || h.chunk_releases != 1)
    mcp_fail(903);
}
