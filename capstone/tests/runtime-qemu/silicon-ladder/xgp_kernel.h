#ifndef XGP_H
#define XGP_H
/* Follow-ups to the xg* minimal repro (2026-08-04). xgs/xgx returned the correct low bits
 * with bit 29 (2^29) set, after storing &globalArr[i] into GLOBAL storage; xgl (same address
 * into a LOCAL) was correct. These arms separate WHAT is corrupted:
 *   xgn  return the bare traversal count n      -> is the COUNT wrong, or only the arithmetic?
 *   xgc  do the stores, return a CONSTANT       -> is the return path poisoned regardless of n?
 *   xgp  return the stored POINTER itself       -> is the STORED VALUE corrupted?
 *   xgr  store, read back, compare pointers     -> store-side or load-side? */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgp_arr[9];
static struct xg_s *xgp_head;
static unsigned xgp_compute(void)
{
  int i;
  xgp_head = 0;
  for (i = 0; i < 9; i++) { xgp_arr[i].v = i; xgp_arr[i].next = xgp_head; xgp_head = &xgp_arr[i]; }
  /* the STORED pointer, as an integer -- compare the board value against QEMU's */
  return (unsigned)(unsigned long)xgp_head;
}
#endif
