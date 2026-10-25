/* LIFETIME CONTROL 3a on block_fast, the successor stays usable. A (the first
 * object of a fresh BLOCK_FAST allocator) is released by wmem_free_all, and
 * wmem is asked for A's size until it hands out A's start address again as C
 * (blockfast.h, wm_bf_successor: printed, and required to be equal). C is
 * written and read back through C.
 *
 * On the protected build C is a child of the NEW generation the reset
 * derived, over the same bytes; the revocation of the old generation must not
 * reach it.
 *
 * Predicted on both builds: C's address equals A's, every access completes.
 * The stale read through A is the separate program 02, so that its fault on
 * the protected build cannot hide or be hidden by this one's result. */
#include "../blockfast.h"

WM_CASE(1) {
  unsigned char *stale;
  (void)wm_bf_successor(&stale);
}
