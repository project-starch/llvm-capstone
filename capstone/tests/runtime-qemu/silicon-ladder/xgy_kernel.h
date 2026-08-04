#ifndef XGY_H
#define XGY_H
/* Separates WALKING the chain from COUNTING it. Board 2026-08-04: xgn (walk + count)
 * returned 9 + 2^27; xgc (stores, NO walk, constant return) and xgr (store + load back +
 * compare, no chained dereference) were both correct. So the remaining variable is the
 * chained dereference `p = p->next` through global storage. */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgy_arr[9];
static struct xg_s *xgy_head;
static unsigned xgy_compute(void)
{
  int i; struct xg_s *p; unsigned long acc = 0;
  xgy_head = 0;
  for (i = 0; i < 9; i++) { xgy_arr[i].v = i; xgy_arr[i].next = xgy_head; xgy_head = &xgy_arr[i]; }
  /* walk, but accumulate the payload ints rather than a counter: is it the COUNTER that is
     poisoned, or any integer live across the chained dereference? sum of v = 0..8 = 36 */
  for (p = xgy_head; p; p = p->next) acc += (unsigned long)p->v;
  return (unsigned)acc;                        /* expect 36 */
}
#endif
