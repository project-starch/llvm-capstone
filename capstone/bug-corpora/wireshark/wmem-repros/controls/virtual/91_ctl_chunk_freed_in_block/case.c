/* CONTROL, not a defect: a chunk the BLOCK allocator frees individually, then read. wmem keeps the
 * chunk on its free list inside a live block, so no system free happens: virtual mallocng never
 * sees this lifetime end (`virtual-malloc` must COMPLETE), while the chunk port revokes the chunk
 * at its own free (`virtual-nested-pools`, mode 1, must FAULT at the read probe). Paired with
 * control 90 (a jumbo the packet-pool reset DOES hand to the system), it tells a column of
 * completions apart from a dead instrument on both arms. The shape is case 12's without the
 * reoccupation: the read follows the free directly. */
#include "corpus.h"

WM_CASE(91) {
  unsigned char *chunk = wmem_alloc(wm_epan_scope(), 16);
  CHECK(chunk, 1);
  memset(chunk, 0x5a, 16);
  wm_held = chunk;
  wmem_free(wm_epan_scope(), chunk); /* an individual BLOCK free: the chunk stays in its block */
  wm_mark();
  WM_READ(wm_held, 0x5a);
}
