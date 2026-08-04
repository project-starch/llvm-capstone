#ifndef XGZ_H
#define XGZ_H
/* Is the trigger the 32-bit access width of the counter slot? xgn's counter is an `int`
 * (lw/sw); xgy's accumulator is an `unsigned long` (ld/sd) and is CORRECT. Same walk, same
 * globals. xgz is xgn with the counter widened to `long` and nothing else changed. */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgz_arr[9];
static struct xg_s *xgz_head;
static unsigned xgz_compute(void)
{
  int i; long n = 0; struct xg_s *p;      /* long, not int -> ld/sd instead of lw/sw */
  xgz_head = 0;
  for (i = 0; i < 9; i++) { xgz_arr[i].v = i; xgz_arr[i].next = xgz_head; xgz_head = &xgz_arr[i]; }
  for (p = xgz_head; p; p = p->next) n++;
  return (unsigned)n;                      /* expect 9 */
}
#endif
