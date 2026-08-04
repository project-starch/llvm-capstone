#ifndef XGS_H
#define XGS_H
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

static struct xg_s xgs_arr[10];        /* slot 9 is the head -- SAME global object */
static unsigned xgs_compute(void)
{
  int i, n = 0; struct xg_s *p;
  xgs_arr[9].next = 0;
  for (i = 0; i < 9; i++) { xgs_arr[i].v = i; xgs_arr[i].next = xgs_arr[9].next; xgs_arr[9].next = &xgs_arr[i]; }
  for (p = xgs_arr[9].next; p; p = p->next) n++;
  return (unsigned)(n * 100 + 63);     /* expect 963 */
}
#endif
