#include "corpus.h"

WM_CASE(0) {
/* Row 0 -- RPC-over-RDMA, #18852 and its report #18910, fix 3c8be14c82.
 * dissect_rpcrdma builds the write-offset array in packet scope and keeps it
 * in the file-scope global gp_rdma_write_offsets; nothing clears the global
 * at frame end. A later packet's process_rdma_list reads the array's count
 * field through it, after the next packet's dissection has reoccupied the
 * retained block. */
  static unsigned char *gp_rdma_write_offsets; /* packet-rpcrdma.c:253 */
  unsigned char *array = wmem_alloc(wm_packet, 80); /* wmem_array_new, :1602 */
  CHECK(array, 1);
  memset(array, 0, 80);
  array[56] = 3; /* the element count, the field the stale read touches */
  gp_rdma_write_offsets = array;
  uintptr_t address = (uintptr_t)array;
  wm_next_packet(); /* wmem_leave_packet_scope at frame end, epan.c:617 */
  /* Packet N+1 takes a path that never re-stores the global; its own first
   * allocation lands on the same storage. Proven before the marker. */
  unsigned char *next = wmem_alloc(wm_packet, 80);
  CHECK(next && (uintptr_t)next == address, 2);
  memset(next, 0x5a, 80);
  /* wmem_array_get_count reads the count field at +56; the case reads the
   * descriptor's first byte, because deriving an interior pointer from
   * revoked authority faults at the arithmetic, before the load. */
  wm_held = gp_rdma_write_offsets; /* :1159 */
  wm_mark();
  (void)wm_probe(wm_held);
}
