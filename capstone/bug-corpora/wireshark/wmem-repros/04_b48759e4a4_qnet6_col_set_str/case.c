#include "corpus.h"

WM_CASE(4) {
/* Row 4 -- qnet6, #19960, fix b48759e4a4.
 * A mass conversion changed col_add_str (copies) to col_set_str (keeps the
 * pointer) on the result of val_to_str, whose default string is not a literal
 * but a wmem_packet_scope() allocation. The scope is torn down at the end of
 * dissection; print_columns then runs strlen over COL_INFO. The fix is on the
 * store side: copy into the column again. */
  unsigned char *info = wmem_alloc(wm_packet, 70); /* val_to_str, packet-qnet6.c:4066 */
  CHECK(info, 1);
  memset(info, 0, 70);
  memcpy(info, "Unknown LWL4 Type 200 packets", 30);
  wm_held = info; /* col_set_str: the pointer, not a copy */
  wm_next_packet(); /* wmem_leave_packet_scope, end of dissection, epan.c:665 */
  wm_mark();
  (void)wm_probe(wm_held); /* strlen in print_columns, tshark.c:4509 */
}
