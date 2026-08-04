#ifndef XGN_H
#define XGN_H
/* Follow-ups to the xg* minimal repro (2026-08-04). xgs/xgx returned the correct low bits
 * with bit 29 (2^29) set, after storing &globalArr[i] into GLOBAL storage; xgl (same address
 * into a LOCAL) was correct. These arms separate WHAT is corrupted:
 *   xgn  return the bare traversal count n      -> is the COUNT wrong, or only the arithmetic?
 *   xgc  do the stores, return a CONSTANT       -> is the return path poisoned regardless of n?
 *   xgp  return the stored POINTER itself       -> is the STORED VALUE corrupted?
 *   xgr  store, read back, compare pointers     -> store-side or load-side? */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgn_arr[9];
static struct xg_s *xgn_head;
static unsigned xgn_compute(void)
{
  int i, n = 0; struct xg_s *p;
  xgn_head = 0;
  for (i = 0; i < 9; i++) { xgn_arr[i].v = i; xgn_arr[i].next = xgn_head; xgn_head = &xgn_arr[i]; }
  for (p = xgn_head; p; p = p->next) n++;
  return (unsigned)n;                    /* expect 9, bare -- no *100 arithmetic */
}
#endif
