#include "corpus.h"

WM_CASE(3) {
/* Row 3 -- SICK CoLA, #20587, fix 6eab9f83ab.
 * dissect_sick_cola_b_pdu reads the command as an ASCII string from
 * wmem_packet_scope() and passes it to col_set_str, which stores the pointer
 * and copies nothing. The scope is torn down at the end of dissection; the
 * print step then runs strlen over COL_INFO for the same packet. */
  unsigned char *command = wmem_alloc(wm_packet, 45); /* tvb_get_string_enc, packet-cola.c:2304 */
  CHECK(command, 1);
  memset(command, 0, 45);
  memcpy(command, "sRA LMDscandata", 16);
  wm_held = command; /* col_set_str(pinfo->cinfo, COL_INFO, ...): the pointer */
  wm_next_packet(); /* wmem_leave_packet_scope, end of dissection, epan.c:680 */
  wm_mark();
  (void)wm_probe(wm_held); /* strlen in print_columns, tshark.c:4630 */
}
