/* Capstone: the authority layer under the ledger, built from Sublet's
 * primitives. The payload arrives as one linear region and is split once,
 * pages below and cache objects above; each unit is carved from its half in
 * address order and never returned, because upstream never returns one either.
 * take mints a copyable alias and keeps the revocation handle in the slot;
 * renew revokes every alias below that handle and mints a fresh one, which is
 * the whole of the protected mode. */
#include "port.h"
#include <sublet/sublet.h>

static capstone_cap_slot remaining[2];
static uintptr_t base[2], cursor[2], end[2];

void mcp_authority_init(void *payload) {
  capstone_cap_store(&remaining[MCP_PAGES], payload);
  uintptr_t bottom = capstone_cap_base(&remaining[MCP_PAGES]);
  if (capstone_cap_type(&remaining[MCP_PAGES]) != CAPSTONE_CAP_LINEAR ||
      capstone_cap_end(&remaining[MCP_PAGES]) != bottom + MCP_PAYLOAD_BYTES ||
      (bottom & (MCP_GRAIN - 1)))
    mcp_fail(501);
  base[MCP_PAGES] = cursor[MCP_PAGES] = bottom;
  end[MCP_PAGES] = bottom + MCP_PAGE_HALF;
  base[MCP_OBJECTS] = cursor[MCP_OBJECTS] = end[MCP_PAGES];
  end[MCP_OBJECTS] = bottom + MCP_PAYLOAD_BYTES;
  sublet_split(&remaining[MCP_PAGES], end[MCP_PAGES], &remaining[MCP_OBJECTS]);
}
int mcp_authority_can_revoke(void) { return 1; }
uintptr_t mcp_authority_base(unsigned half) { return base[half]; }
size_t mcp_authority_used(unsigned half) { return cursor[half] - base[half]; }
int mcp_authority_carve(unsigned half, size_t size, capstone_cap_slot *out, uintptr_t *at) {
  if (half > 1 || cursor[half] > end[half] - size)
    return 0;
  *at = cursor[half];
  cursor[half] += size;
  sublet_carve(&remaining[half], cursor[half], out);
  return 1;
}
void mcp_authority_split(capstone_cap_slot *from, uintptr_t split_end, capstone_cap_slot *out) {
  sublet_carve(from, split_end, out);
}
void *mcp_authority_take(capstone_cap_slot *region) { return sublet_take(region); }
void *mcp_authority_renew(capstone_cap_slot *region) {
  sublet_give(region);
  return sublet_take(region);
}
void mcp_authority_reclaim(capstone_cap_slot *region) { sublet_give(region); }
