#include "corpus.h"

WM_CASE(5) {
/* Row 5 -- USBLL, #17367 and its duplicate #17368, fix 5a109265a6.
 * usbll_set_address allocates the three-byte address structure from
 * wmem_packet_scope() and hands it to set_address, which keeps the pointer in
 * pinfo->src. That scope is torn down at the end of dissection; column fill
 * for the same packet then calls usbll_addr_to_str, which reads the flags. */
  unsigned char *src_addr = wmem_alloc(wm_packet, 3); /* wmem_new0, packet-usbll.c:662 */
  CHECK(src_addr, 1);
  src_addr[0] = 0x02; /* flags */
  src_addr[1] = 7;    /* device */
  src_addr[2] = 1;    /* endpoint */
  wm_held = src_addr; /* set_address(&pinfo->net_src, ..., src_addr), :698 */
  wm_next_packet();   /* wmem_leave_packet_scope, end of dissection, epan.c:611 */
  wm_mark();
  (void)wm_probe(wm_held); /* addrp->flags in usbll_addr_to_str, :629 */
}
