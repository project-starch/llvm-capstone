#include "corpus.h"

WM_CASE(12) {
/* Row 12 -- XML, #20664, fix 90bb3a5c9e.
 * register_dtd registers a protocol whose name is the DTD's root name, an
 * epan-scope string that proto_register_protocol stores by pointer, then
 * frees that string with wmem_free at the end of registration. The recycler
 * puts the chunk on its free list inside the live block. Every later packet
 * that adds the protocol item reads hfinfo->name through g_strdup.
 *
 * This is the one case in the corpus whose lifetime ends by an INDIVIDUAL
 * free in the block allocator, not by a pool reset. Sublet lends whole
 * regions; a chunk inside a live block has none of its own, so no epoch ends
 * here and both arms are expected to complete. The row is kept because
 * hiding a known non-detection would misstate what the mechanism covers. */
  static const unsigned char *hfinfo_name;         /* protocol->name / hfinfo->name */
  unsigned char *root_name = wmem_alloc(wm_epan_scope(), 16); /* packet-xml.c:1453 */
  CHECK(root_name, 1);
  memset(root_name, 0, 16);
  memcpy(root_name, "presentation", 13);
  hfinfo_name = root_name;                          /* proto_register_protocol stores the pointer, :1616 */
  uintptr_t address = (uintptr_t)root_name;
  wmem_free(wm_epan_scope(), root_name);            /* :1631, the line the fix removes */
  /* The recycler hands the same chunk to the next request of that size, so
   * the registry's name now aliases whatever is stored there. Proven before
   * the marker. */
  unsigned char *next = wmem_alloc(wm_epan_scope(), 16);
  CHECK(next && (uintptr_t)next == address, 2);
  memset(next, 0x5a, 16);
  wm_next_packet(); /* packets go by; the registry outlives them all */
  wm_held = (unsigned char *)hfinfo_name;
  wm_mark();
  (void)wm_probe(wm_held); /* g_strdup(hfinfo->name) in value_set, ftype-protocol.c:54 */
}
