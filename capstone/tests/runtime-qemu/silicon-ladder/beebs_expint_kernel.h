#ifndef BEEBS_EXPINT_KERNEL_H
#define BEEBS_EXPINT_KERNEL_H
/* Silicon-ladder rung: BEEBS expint -- exponential integral, verbatim compute.
 *
 * Source: Bristol/Embecosm BEEBS `expint` (from Numerical Recipes). Both legs of
 * the branch, the nested psi loop and the foo() helper are unmodified.
 *
 * WHY THIS ONE. It is selected against R-1's shape, not for benchmark prestige.
 * R-1 needs a load through one capability register while a store through another
 * into the same object is pending; this kernel has **no arrays at all** -- every
 * value lives in a scalar local, and the only global is the volatile accumulator
 * the build gate requires. A static screen of the upstream source finds zero
 * array stores inside any loop.
 *
 * Track record this selection has to beat: 6 rungs attempted on silicon today, 2
 * passed. The four failures each turned out to contain a same-object
 * load-with-intervening-store, including rv8_sha512, where I predicted PASS after
 * checking only the read-only table and missing that sha_w[i&15] is written in the
 * compression loop. So the screen here is on the WHOLE kernel, not the part that
 * caught my eye.
 *
 * The residual risk is real and worth stating: beebs_fibcall is also pure scalar
 * and still miscomputed on silicon, so "no arrays" has not been sufficient before.
 * If this fails too, that is evidence R-1 is not the only mechanism -- which is
 * already R-6's standing message. */

static volatile long ei_sink;   /* satisfies the ldc gp[i] build gate */

static long ei_foo(long x) { return (x * x + (8 * x)) << (4 - x); }

static long ei_expint(int n, long x) {
  int i, ii, nm1;
  long a, b, c, d, del, fact, h, psi, ans;
  nm1 = n - 1;
  if (x > 1) {
    b = x + n;
    c = 2e6;
    d = 3e7;
    h = d;
    ans = 0;
    for (i = 1; i <= 100; i++) {
      a = -i * (nm1 + i);
      b += 2;
      d = 10 * (a * d + b);
      c = b + a / c;
      del = c * d;
      h *= del;
      if (del < 10000) { ans = h * -x; return ans; }
    }
  } else {
    ans = nm1 != 0 ? 2 : 1000;
    fact = 1;
    for (i = 1; i <= 100; i++) {
      fact *= -x / i;
      if (i != nm1) {
        del = -fact / (i - nm1);
      } else {
        psi = 0x00FF;
        for (ii = 1; ii <= nm1; ii++)
          psi += ii + nm1;
        del = psi + fact * ei_foo(x);
      }
      ans += del;
    }
  }
  return ans;
}

static unsigned expint_compute(void) {
  unsigned h = 2166136261u;
  for (int rep = 0; rep < 64; rep++) {
    ei_sink = ei_expint(50, (long)(rep & 1));   /* upstream's n=50 argument */
    h ^= (unsigned)ei_sink;        h *= 16777619u;
    h ^= (unsigned)(ei_sink >> 32); h *= 16777619u;
  }
  return h;
}
#endif
