/* Wireshark #16807 — Lua TvbRange userdata ⟷ C tvbuff use-after-free.
 * Source: ../../wireshark-16807/boundary.md. valgrind: invalid READ size 1 and
 * 4, 16 bytes inside a 72-byte tvbuff freed by tvb_free_chain.
 *
 * Two allocations: the TvbRange userdata (stashed in the global
 * ProtocolState[id][field], so it SURVIVES GC) and the C tvbuff
 * (tvb_new_subset_remaining).
 *   Free-site (epan/tvbuff.c): epan_dissect_reset -> tvb_free_chain frees the
 *     packet's tvbuff chain between the two analysis passes. Native-frees.
 *   Stale-use: re-dissection reuses the stashed stale TvbRange ->
 *     subtree:add(foo_field, staleRange) -> tvb_ensure_bytes_exist reads the
 *     freed tvbuff's fields.
 * READ at OFFSET 16 -> interior address via cincoffset on the revoked
 * capability (assert-on-untagged FAULT route). Control: the read returns; MISS.
 */
#include "luac_shim.h"
#include <stdint.h>

#define TVBUFF_BYTES 72
#define TVB_FIELD_OFF 16 /* the tvbuff field valgrind names */

static volatile uint64_t sink;

int main(void) {
  unsigned char *tvb = (unsigned char *)malloc(TVBUFF_BYTES); /* tvb_new_subset */
  if (!tvb)
    abort();
  memset(tvb, 0, TVBUFF_BYTES);

  /* The TvbRange userdata stashed in ProtocolState survives GC and keeps the
   * borrowed tvbuff pointer. */
  unsigned char *stale_range = tvb;

  free(tvb); /* epan_dissect_reset -> tvb_free_chain -> REVOKE */

  /* Re-dissection: subtree:add(staleRange) -> tvb_ensure_bytes_exist reads a
   * tvbuff field at offset 16. */
  sink = *(volatile uint32_t *)(stale_range + TVB_FIELD_OFF); /* tvb_ensure_bytes_exist */

  mock_report("luac_tvbuff_uaf", "use-after-free-survived");
  return 0;
}
