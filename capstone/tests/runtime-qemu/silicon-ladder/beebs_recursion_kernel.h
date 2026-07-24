#ifndef BEEBS_RECURSION_KERNEL_H
#define BEEBS_RECURSION_KERNEL_H
/* Silicon-ladder rung 4: BEEBS recursion (a *found* benchmark, kept faithful).
 *
 * Source: Bristol/Embecosm BEEBS `recursion`. Exercises what the earlier
 * (iterative) rungs did not: DEEP self-recursion (fib) and MUTUAL recursion
 * (anka<->kalle) -- i.e. gp-free plain call/ret under a tall, reentrant call
 * stack. Two globals: `volatile int In` (a volatile-global store) and a
 * `static int n`. Single-TU, integer. Checksum folds fib(n), anka(n), kalle(n)
 * so both recursion styles contribute; a native host folds the identical value. */

static volatile int In;
static int rn;

static int fib(int i) {
  if (i == 0) return 1;
  if (i == 1) return 1;
  return fib(i - 1) + fib(i - 2);
}

static int anka(int);            /* mutual recursion */
static int kalle(int i) { return (i <= 0) ? 0 : anka(i - 1); }
static int anka(int i)  { return (i <= 0) ? 1 : kalle(i - 1); }

static unsigned rec_compute(void) {
  rn = 10;
  In = fib(rn);                  /* volatile-global store, In == 89 */
  int vals[3];
  vals[0] = In;
  vals[1] = anka(rn);
  vals[2] = kalle(rn);
  unsigned h = 2166136261u;      /* FNV-1a over the three results, LE bytes */
  for (int t = 0; t < 3; t++) {
    unsigned v = (unsigned)vals[t];
    for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
  }
  return h;
}
#endif
