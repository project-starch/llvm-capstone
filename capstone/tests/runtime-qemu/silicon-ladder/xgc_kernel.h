#ifndef XGC_H
#define XGC_H
/* Follow-ups to the xg* minimal repro (2026-08-04). xgs/xgx returned the correct low bits
 * with bit 29 (2^29) set, after storing &globalArr[i] into GLOBAL storage; xgl (same address
 * into a LOCAL) was correct. These arms separate WHAT is corrupted:
 *   xgn  return the bare traversal count n      -> is the COUNT wrong, or only the arithmetic?
 *   xgc  do the stores, return a CONSTANT       -> is the return path poisoned regardless of n?
 *   xgp  return the stored POINTER itself       -> is the STORED VALUE corrupted?
 *   xgr  store, read back, compare pointers     -> store-side or load-side? */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgc_arr[9];
static struct xg_s *xgc_head;
static unsigned xgc_compute(void)
{
  int i;
  xgc_head = 0;
  for (i = 0; i < 9; i++) { xgc_arr[i].v = i; xgc_arr[i].next = xgc_head; xgc_head = &xgc_arr[i]; }
  return 777u;                           /* stores happen; return ignores them entirely */
}
#endif
