/* CONTROL, not a defect: one packet-pool chunk, the packet reset, then a read
 * through the chunk's alias. The reset keeps the block_fast block, so the
 * system allocator sees nothing: the read COMPLETES unless wmem's own objects
 * are lifetimes (WM_SUBLET), where the reset revoked it and the read FAULTS. */
#include "corpus.h"

WM_CASE(0) {
  unsigned char *chunk = wmem_alloc(wm_packet, 64);
  CHECK(chunk, 1);
  memset(chunk, 0x5a, 64);
  wm_held = chunk;
  wm_next_packet(); /* wmem_free_all on the packet pool: the first block is kept */
  wm_mark();
  WM_READ(wm_held, 0x5a);
}
