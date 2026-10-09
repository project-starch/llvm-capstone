#include "corpus.h"

WM_CASE(2) {
/* Row 2 -- MDB, #21261, fix 99da8c2cdc.
 * mdb_set_addrs formats an unknown peripheral address with val_to_str, whose
 * default string comes from wmem_packet_scope(), and hands it to set_address
 * as AT_STRINGZ, which keeps the pointer. That scope is torn down at the end
 * of dissection, before tshark prints the packet; print_columns then runs
 * strlen over the address column. No later allocation intervenes, so the
 * unprotected read returns the old bytes and nothing notices. */
  /* THE FIX, 99da8c2cdc: the column string is allocated where it outlives the print step
   * (pinfo->pool, torn down after printing) -- modelled as a scope the packet reset does not end. */
  unsigned char *periph = wmem_alloc(wm_fixed ? wm_file_scope() : wm_packet, 55); /* val_to_str, packet-mdb.c:257 */
  CHECK(periph, 1);
  memset(periph, 0, 55);
  memcpy(periph, "Unknown (0x2a)", 15);
  wm_held = periph; /* set_address(&pinfo->dst, AT_STRINGZ, ..., periph), :263 */
  wm_next_packet(); /* wmem_leave_packet_scope, end of dissection, epan.c:668 */
  wm_reoccupy(periph, 55); /* native observer only: the next dissection takes the freed storage */
  wm_mark();
  WM_READ(wm_held, WM_MARKER); /* strlen in print_columns, tshark.c:4597 */
}
