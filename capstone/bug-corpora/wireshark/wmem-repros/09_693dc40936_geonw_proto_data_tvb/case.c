#include "corpus.h"

/* The proto-data entry GeoNW adds, as far as this case needs it: the stored
 * value is the tvb, whose backing buffer is what the later read touches. */
struct proto_data_entry {
  unsigned char *tvb_real_data;
};

WM_CASE(9) {
/* Row 9 -- GeoNetworking over PPP, #18779, fix 693dc40936.
 * remove_escape_chars unescapes the frame into a pinfo->pool buffer and wraps
 * it in a child tvb. GeoNW stashes that tvb in proto data added with
 * wmem_file_scope(), so the entry outlives the packet. The pool is reset
 * between packets; a later frame retrieves the entry and reads the header
 * type through the stale tvb. */
  struct proto_data_entry *entry =
      wmem_alloc(wm_file_scope(), sizeof *entry); /* p_add_proto_data(wmem_file_scope(), ...), packet-geonw.c:1913 */
  CHECK(entry, 1);
  unsigned char *buff = wmem_alloc(wm_packet, 55); /* wmem_alloc(pinfo->pool, length), packet-ppp.c:5914 */
  CHECK(buff, 2);
  memset(buff, 0x7e, 55);
  buff[1] = 0x20; /* the header type byte the later read wants */
  entry->tvb_real_data = buff; /* tvb_new_child_real_data(tvb, buff, ...) then stored */
  wm_next_packet(); /* epan_dissect_reset, epan.c:581 */
  /* A later frame's dissect_geonw retrieves the entry, :2229-2231. */
  /* tvb_get_guint8(tvb, 1) reads byte 1; the case reads byte 0 through the
   * retrieved pointer, because deriving an interior pointer from revoked
   * authority faults at the arithmetic, before the load. */
  wm_held = entry->tvb_real_data;
  wm_mark();
  (void)wm_probe(wm_held); /* :2234 */
}
