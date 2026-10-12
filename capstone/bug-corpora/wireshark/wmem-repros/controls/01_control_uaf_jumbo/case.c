/* CONTROL, not a defect: the one sequence under which a stock wmem DOES give a
 * packet-scoped object back to the system before a later read. A request larger
 * than a BLOCK_FAST block becomes a jumbo with its own g_malloc, and
 * wmem_free_all frees every jumbo with g_free (wmem_allocator_block_fast.c,
 * wmem_block_fast_free_all). A system allocator that retires what it frees --
 * virtual mallocng, ASan, CheriBSD's revocation once it sweeps -- sees this
 * free, so the stale read must FAULT (or be reported) on every arm. */
#include "corpus.h"

#define JUMBO (3UL << 20) /* past WMEM_BLOCK_MAX_ALLOC_SIZE of a 2 MiB fast block */

WM_CASE(1) {
  unsigned char *jumbo = wmem_alloc(wm_packet, JUMBO);
  CHECK(jumbo, 1);
  memset(jumbo, 0x5a, 64);
  wm_held = jumbo;
  wm_next_packet(); /* wmem_free_all on the packet pool: every jumbo goes to g_free */
  wm_mark();
  WM_READ(wm_held, 0x5a);
}
