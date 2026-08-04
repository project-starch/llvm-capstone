#ifndef XGW_H
#define XGW_H
/* Separates WALKING the chain from COUNTING it. Board 2026-08-04: xgn (walk + count)
 * returned 9 + 2^27; xgc (stores, NO walk, constant return) and xgr (store + load back +
 * compare, no chained dereference) were both correct. So the remaining variable is the
 * chained dereference `p = p->next` through global storage. */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgw_arr[9];
static struct xg_s *xgw_head;
static unsigned xgw_compute(void)
{
  int i; struct xg_s *p; int seen = 0;
  xgw_head = 0;
  for (i = 0; i < 9; i++) { xgw_arr[i].v = i; xgw_arr[i].next = xgw_head; xgw_head = &xgw_arr[i]; }
  for (p = xgw_head; p; p = p->next) seen++;   /* WALK, and the count is DISCARDED */
  if (seen == 9) return 777u;                  /* constant return after a full walk */
  return 555u;                                 /* distinct: the walk ran a wrong number of times */
}
#endif
