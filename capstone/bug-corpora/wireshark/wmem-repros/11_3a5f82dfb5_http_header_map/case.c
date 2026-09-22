#include "corpus.h"

/* The conversation's per-request private data, as far as this case needs it. */
struct http_req_res_private_data {
  unsigned char *request_headers;
  unsigned char *response_headers;
};

WM_CASE(11) {
/* Row 11 -- HTTP, #20702 and its duplicate #20703, fix 3a5f82dfb5.
 * A header line seen before any start line makes dissect_http_message create
 * its header map in pinfo->pool; when the start line follows in the same
 * message, that map is saved into the conversation's file-scope private data
 * instead of a file-scope one. The pool is reset between packets, another
 * packet's allocation reoccupies the storage, and a later frame retrieves the
 * map and calls wmem_map_insert on it, which begins by reading map->table. */
  struct http_req_res_private_data *prv_data =
      wmem_alloc(wm_file_scope(), sizeof *prv_data); /* wmem_new0(wmem_file_scope()), packet-http.c:1124 */
  CHECK(prv_data, 1);
  unsigned char *map = wmem_alloc(wm_packet, 88); /* wmem_map_new(pinfo->pool, ...), :1814 */
  CHECK(map, 2);
  memset(map, 0, 88);
  prv_data->request_headers = map; /* :1781 */
  uintptr_t address = (uintptr_t)map;
  wm_next_packet(); /* epan_dissect_reset, epan.c:625 */
  /* Another packet's dissection reoccupies the storage: the report's own
   * allocator log shows an unrelated allocation sitting in the freed block.
   * Proven before the marker. */
  unsigned char *other = wmem_alloc(wm_packet, 88);
  CHECK(other && (uintptr_t)other == address, 3);
  memset(other, 0x5a, 88);
  /* A later frame retrieves the map from the conversation, :1797-1804. */
  /* map->table sits at +16; the case reads the struct's first byte, because
   * deriving an interior pointer from revoked authority faults at the
   * arithmetic, before the load. */
  wm_held = prv_data->request_headers; /* wmem_map.c:299 */
  wm_mark();
  (void)wm_probe(wm_held);
}
