#ifndef XGR_H
#define XGR_H
/* Follow-ups to the xg* minimal repro (2026-08-04). xgs/xgx returned the correct low bits
 * with bit 29 (2^29) set, after storing &globalArr[i] into GLOBAL storage; xgl (same address
 * into a LOCAL) was correct. These arms separate WHAT is corrupted:
 *   xgn  return the bare traversal count n      -> is the COUNT wrong, or only the arithmetic?
 *   xgc  do the stores, return a CONSTANT       -> is the return path poisoned regardless of n?
 *   xgp  return the stored POINTER itself       -> is the STORED VALUE corrupted?
 *   xgr  store, read back, compare pointers     -> store-side or load-side? */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgr_arr[9];
static struct xg_s *xgr_slot[9];         /* GLOBAL array of pointers */
static unsigned xgr_compute(void)
{
  int i, ok = 0;
  for (i = 0; i < 9; i++) { xgr_arr[i].v = i; xgr_slot[i] = &xgr_arr[i]; }
  /* read back and compare against a freshly-formed address: store-side or load-side? */
  for (i = 0; i < 9; i++) if (xgr_slot[i] == &xgr_arr[i]) ok++;
  return (unsigned)(ok * 10 + 7);        /* expect 97 */
}
#endif
