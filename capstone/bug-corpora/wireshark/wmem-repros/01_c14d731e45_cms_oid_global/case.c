#include "corpus.h"

WM_CASE(1) {
/* Row 1 -- CMS, #17800/#17809/#17835/#17935, fix c14d731e45.
 * dissect_cms_T_capability decodes an OID into a packet-scope string and
 * keeps the pointer in the file-scope global object_identifier_id. When a
 * later packet's capability decode fails non-fatally, T_parameters reads the
 * global from the earlier packet: call_ber_oid_callback ends in g_strdup, a
 * strlen over storage the next dissection has already reoccupied. */
  static const unsigned char *object_identifier_id; /* packet-cms.c:314 */
  unsigned char *oid = wmem_alloc(wm_packet, 59); /* wmem_strbuf_finalize, cms.cnf:210 */
  CHECK(oid, 1);
  memset(oid, 0, 59);
  memcpy(oid, "1.2.840.113549.1.9.16.3.6", 25);
  object_identifier_id = oid;
  uintptr_t address = (uintptr_t)oid;
  wm_next_packet(); /* wmem_leave_packet_scope at frame end, epan.c:617 */
  /* The next packet's decode raises before it re-sets the global; its other
   * allocations reoccupy the block. Proven before the marker. */
  unsigned char *next = wmem_alloc(wm_packet, 59);
  CHECK(next && (uintptr_t)next == address, 2);
  memset(next, 0x5a, 59);
  wm_held = (unsigned char *)object_identifier_id; /* T_parameters, cms.cnf:220 */
  wm_mark();
  (void)wm_probe(wm_held); /* g_strdup -> strlen in find_string_dtbl_entry */
}
