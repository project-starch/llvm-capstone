/* Hosted: the same layer with plain pointers and no authority. A slot holds
 * the address a unit was carved at, take hands it back, and a unit keeps that
 * address for the whole run -- which is exactly what upstream gives every
 * consumer, so the native arms measure the allocator as shipped. renew is
 * refused rather than faked: a native execution does not acquire revocation
 * from a mode number, and leases.c asks before accepting mode 1. */
#include "port.h"

static unsigned char *base[2];
static size_t used[2];
static const size_t limit[2] = {MCP_PAGE_HALF, MCP_OBJECT_HALF};

void mcp_authority_init(void *payload) {
  if ((uintptr_t)payload & (MCP_GRAIN - 1))
    mcp_fail(501);
  base[MCP_PAGES] = payload;
  base[MCP_OBJECTS] = base[MCP_PAGES] + MCP_PAGE_HALF;
  used[MCP_PAGES] = used[MCP_OBJECTS] = 0;
}
int mcp_authority_can_revoke(void) { return 0; }
uintptr_t mcp_authority_base(unsigned half) { return (uintptr_t)base[half]; }
size_t mcp_authority_used(unsigned half) { return used[half]; }
int mcp_authority_carve(unsigned half, size_t size, capstone_cap_slot *out, uintptr_t *at) {
  if (half > 1 || used[half] > limit[half] - size)
    return 0;
  out->c = base[half] + used[half];
  *at = (uintptr_t)out->c;
  used[half] += size;
  return 1;
}
void mcp_authority_split(capstone_cap_slot *from, uintptr_t split_end, capstone_cap_slot *out) {
  out->c = from->c;
  from->c = (void *)split_end;
}
void *mcp_authority_take(capstone_cap_slot *region) { return region->c; }
void *mcp_authority_renew(capstone_cap_slot *region) {
  (void)region;
  mcp_fail(505);
}
void mcp_authority_reclaim(capstone_cap_slot *region) { (void)region; }
