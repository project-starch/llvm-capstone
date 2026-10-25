/* LIFETIME CONTROL 3a, the successor stays usable. A is released with
 * wmem_free, and wmem is asked for A's size until it hands out A's start
 * address again as C (lifetime.h, wm_ctl_successor: printed, and required to
 * be equal). C is written and read back through C.
 *
 * On the protected build C is a NEW CDERIVE child of the block's generation
 * over the same bytes; the revocation of A must not reach it.
 *
 * Predicted on both builds: C's address equals A's, every access completes.
 * The stale read through A is the separate program 03, so that its fault on
 * the protected build cannot hide or be hidden by this one's result. */
#include "../lifetime.h"

WM_CASE(2) {
  unsigned char *stale;
  (void)wm_ctl_successor(&stale);
}
