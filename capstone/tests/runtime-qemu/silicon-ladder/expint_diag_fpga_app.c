/* R-8 bisect: expint runs to completion on silicon and returns a WRONG value, in a
 * kernel with NO arrays -- so R-1 (a same-object memory hazard) cannot explain it.
 * This rung returns the intermediates RAW through the debug slots instead of a
 * checksum, because a checksum cannot say WHICH step diverged. Same lesson as
 * insertsort_diag, which is what localised R-1 in the first place.
 *
 * Every slot is scalar arithmetic on locals. If a slot differs from the native
 * value, the operation feeding it is the one that miscomputes on this silicon.
 *
 * Layout (n = 50, so nm1 = 49; x = 0, the branch the benchmark actually takes):
 *   dbg0  branch taken            1 = (x>1) leg, 0 = else leg
 *   dbg1  ans initial             nm1 != 0 ? 2 : 1000
 *   dbg2  fact after i=1          fact *= -x/i
 *   dbg3  fact after the loop
 *   dbg4  psi                     0xFF + sum(ii + nm1), ii = 1..nm1  -- nested loop
 *   dbg5  ei_foo(x)               (x*x + 8x) << (4 - x)  -- the shift
 *   dbg6  del at i == nm1         psi + fact*foo(x)
 *   dbg7  ans final
 *   dbg8  loop trip count         must be 100
 *   dbg9  sum(ii) alone           isolates the nested loop's adder from psi
 */
static volatile long eid_sink;

static long eid_foo(long x) { return (x * x + (8 * x)) << (4 - x); }

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  const int n = 50;
  long x = 0;
  __asm__ volatile("" : "+r"(x));      /* opaque: no constant folding of the whole thing */

  int i, ii, nm1 = n - 1;
  long del = 0, fact, psi = 0, ans, foo_v = 0, del_at_nm1 = 0;
  long fact_i1 = 0, sum_ii = 0;
  int trips = 0;

  res[3 + 0] = (unsigned long)(x > 1);
  ans = nm1 != 0 ? 2 : 1000;
  res[3 + 1] = (unsigned long)ans;

  fact = 1;
  for (i = 1; i <= 100; i++) {
    trips++;
    fact *= -x / i;
    if (i == 1) fact_i1 = fact;
    if (i != nm1) {
      del = -fact / (i - nm1);
    } else {
      psi = 0x00FF;
      for (ii = 1; ii <= nm1; ii++) {
        psi += ii + nm1;
        sum_ii += ii;
      }
      foo_v = eid_foo(x);
      del = psi + fact * foo_v;
      del_at_nm1 = del;
    }
    ans += del;
  }

  res[3 + 2] = (unsigned long)fact_i1;
  res[3 + 3] = (unsigned long)fact;
  res[3 + 4] = (unsigned long)psi;
  res[3 + 5] = (unsigned long)foo_v;
  res[3 + 6] = (unsigned long)del_at_nm1;
  res[3 + 7] = (unsigned long)ans;
  res[3 + 8] = (unsigned long)trips;
  res[3 + 9] = (unsigned long)sum_ii;

  eid_sink = ans;
  res[0] = (unsigned)ans;
  res[2] = 0xD09E;
}
