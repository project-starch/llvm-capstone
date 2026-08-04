/* lua-curl #80 — easy userdata ⟷ multi userdata back-pointer use-after-free.
 * Source: ../../lua-curl-80/boundary.md. ASan: heap-use-after-free WRITE size 8,
 * 64 bytes inside a 96-byte lcurl_multi_t userdata freed by Lua's GC.
 *
 * Two allocations: the easy userdata (lcurl_easy_t, carries a raw back-pointer
 * multi set by multi:add_handle) and the multi userdata (lcurl_multi_t, 96-byte
 * Lua-GC block wrapping a native CURLM*).
 *   Free-site: m = nil; collectgarbage() makes the multi unreachable ->
 *     lcurl_multi_cleanup -> Lua frees the 96-byte lcurl_multi_t via l_alloc.
 *     The vulnerable tree does NOT null e->multi on the still-attached easy.
 *   Stale-use (lceasy.c:87): easy:close() -> lcurl_easy_cleanup runs
 *     if(p->multi){ p->multi->L = L; } -> a WRITE through the dangling back-ptr.
 * WRITE size 8 at OFFSET 64 (the ->L field) -> interior store through the
 * revoked capability (assert-on-untagged FAULT route). Control: the store
 * completes and the row reports MISS.
 *
 * CAVEAT (spare-tier, from the case): the freed block is Lua-managed userdata
 * memory, not a curl-owned native heap object — analogous to luv-503. The
 * allocator-visible event is identical.
 */
#include "luac_shim.h"
#include <stdint.h>

#define LCURL_MULTI_BYTES 96
#define MULTI_L_OFF 64 /* the ->L field ASan names */

int main(void) {
  unsigned char *multi = (unsigned char *)malloc(LCURL_MULTI_BYTES); /* multi userdata */
  if (!multi)
    abort();
  memset(multi, 0, LCURL_MULTI_BYTES);

  /* multi:add_handle sets easy->multi to a raw back-pointer at the multi. */
  unsigned char *easy_multi = multi;

  free(multi); /* m=nil; collectgarbage() -> l_alloc frees the userdata -> REVOKE */

  /* easy:close -> if(p->multi){ p->multi->L = L; } -> write ->L at offset 64. */
  *(volatile uint64_t *)(easy_multi + MULTI_L_OFF) = 0xA5A5A5A5UL; /* lceasy.c:87 stores L */

  mock_report("luac_curl_multi_backptr_uaf", "use-after-free-survived");
  return 0;
}
