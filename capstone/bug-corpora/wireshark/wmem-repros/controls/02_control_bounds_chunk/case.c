/* CONTROL, not a defect: a read one byte past a live 24-byte packet-pool
 * chunk, inside the same block_fast block. The byte belongs to the block the
 * system allocator handed out, so the read COMPLETES unless wmem bounds its
 * own objects (WM_SUBLET), where it FAULTS on the chunk's bound. */
#include "corpus.h"

WM_CASE(2) {
  unsigned char *chunk = wmem_alloc(wm_packet, 24);
  CHECK(chunk, 1);
  memset(chunk, 0x5a, 24);
  wm_held = chunk + 24;
  wm_mark();
  WM_READ_AT(wm_held, chunk, 24);
}
