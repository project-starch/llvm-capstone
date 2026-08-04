#ifndef XGX_H
#define XGX_H
/* Minimal repro candidate for the SQLite stage-10 wedge, bisected 2026-08-04 to
 * sqlite3InsertBuiltinFuncs. Its inner loop stores the ADDRESS OF AN ELEMENT OF ONE GLOBAL
 * into a DIFFERENT global:
 *     aDef[i].u.pHash = sqlite3BuiltinFunctions.a[h];
 *     sqlite3BuiltinFunctions.a[h] = &aDef[i];
 * Under -capstone-gp-captable every global is its own carved capability, so this is
 * CROSS-GLOBAL capability storage. Every probe that passed (stages 11-16, 18) stored string
 * literals; none stored the address of a global array element into another global.
 *
 * Three arms separate the two candidate triggers:
 *   xgl  &global[i] -> a LOCAL array          (taking the address, no cross-global store)
 *   xgs  &global[i] -> the SAME global object  (cross-element, but one capability)
 *   xgx  &global[i] -> a DIFFERENT global      (the real construct)
 * Distinct sentinels so a mis-selected arm cannot read as a passing one. */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgx_arr[9];
static struct xg_s *xgx_head;          /* a DIFFERENT global -- the real construct */
static unsigned xgx_compute(void)
{
  int i, n = 0; struct xg_s *p;
  xgx_head = 0;
  for (i = 0; i < 9; i++) { xgx_arr[i].v = i; xgx_arr[i].next = xgx_head; xgx_head = &xgx_arr[i]; }
  for (p = xgx_head; p; p = p->next) n++;
  return (unsigned)(n * 100 + 61);     /* expect 961 */
}
#endif
