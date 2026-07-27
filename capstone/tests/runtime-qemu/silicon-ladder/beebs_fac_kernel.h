#ifndef BEEBS_FAC_KERNEL_H
#define BEEBS_FAC_KERNEL_H
/* Silicon-ladder rung: BEEBS fac -- plain single recursion.
 *
 * Source: Bristol/Embecosm BEEBS `fac`. Verbatim compute.
 *
 * SHAPE PREDICTION under issue R-1 (ref/ISSUES.md): PASS. No arrays; the only
 * memory traffic is the call frames.
 *
 * Why it earns a row, and it is the most valuable of the four: the measured
 * table's largest number by far is `beebs_recursion` at 1.801x cycles, and that
 * headline currently rests on ONE benchmark. `beebs_recursion` is deep AND
 * mutual recursion; `fac` is shallow, single, self-recursion. If fac also lands
 * near 1.8x, the cost is the gp-free call/return ABI per call, which is what we
 * claim. If fac lands near 1.05x, the 1.8x is about recursion DEPTH (frame
 * pressure, capability spills) and the claim has to be rewritten. Either
 * outcome fixes a real weakness in the evaluation.
 *
 * `fac_s` / `fac_n` are `volatile` exactly as upstream declares them, and that
 * is load-bearing here: a rung with no globals emits no `ldc gp[i]`, never
 * touches the gp cap-table, and so would be priced without the ABI that is the
 * subject of the measurement. The ladder build gate rejects such a rung. */

static volatile int fac_s;
static volatile int fac_n;

static int fac_fac(int n) {
  if (n == 0)
    return 1;
  else
    return (n * fac_fac(n - 1));
}

static unsigned fac_compute(void) {
  unsigned h = 2166136261u;
  for (int rep = 0; rep < 256; rep++) {
    fac_s = 0;
    fac_n = 10;                        /* upstream's bound */
    for (int i = 0; i <= fac_n; i++)
      fac_s += fac_fac(i);
    h ^= (unsigned)fac_s;
    h *= 16777619u;
  }
  return h;
}
#endif
