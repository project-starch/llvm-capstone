#ifndef XGQ_H
#define XGQ_H
/* Is the trigger the 32-bit access width of the counter slot? xgn's counter is an `int`
 * (lw/sw); xgy's accumulator is an `unsigned long` (ld/sd) and is CORRECT. Same walk, same
 * globals. xgz is xgn with the counter widened to `long` and nothing else changed. */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgq_arr[9];
static struct xg_s *xgq_head;
static unsigned xgq_compute(void)
{
  int i, n = 0; struct xg_s *p;
  xgq_head = 0;
  for (i = 0; i < 9; i++) { xgq_arr[i].v = i; xgq_arr[i].next = xgq_head; xgq_head = &xgq_arr[i]; }
  for (p = xgq_head; p; p = p->next) n++;
  /* int counter as in xgn, but MASKED to 16 bits: if only the high bits are poisoned this
     returns the correct 9, which localises the damage to bits above the value. */
  return (unsigned)(n & 0xffff);           /* expect 9 */
}
#endif
