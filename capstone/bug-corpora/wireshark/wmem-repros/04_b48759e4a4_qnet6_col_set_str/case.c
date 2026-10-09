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
  if (wm_fixed) {
    /* THE FIX, b48759e4a4: col_add_str again -- the column COPIES the string into storage it owns,
     * which the packet reset does not end. */
    unsigned char *column = wmem_alloc(wm_file_scope(), 70);
    CHECK(column, 2);
    memcpy(column, info, 70);
    wm_held = column;
  }
  wm_next_packet(); /* wmem_leave_packet_scope, end of dissection, epan.c:665 */
  wm_reoccupy(info, 70); /* native observer only */
  wm_mark();
  WM_READ(wm_held, WM_MARKER); /* strlen in print_columns, tshark.c:4509 */
}
