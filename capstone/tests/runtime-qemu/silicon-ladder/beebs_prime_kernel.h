#ifndef BEEBS_PRIME_KERNEL_H
#define BEEBS_PRIME_KERNEL_H
/* Silicon-ladder rung 5: BEEBS prime (a *found* benchmark, kept faithful).
 *
 * Source: Bristol/Embecosm BEEBS `prime`. Integer primality testing over a real
 * call graph (prime->even->divides). Distinct coverage: `swap(&x,&y)` takes the
 * ADDRESS of two globals and dereferences through the pointers -- i.e. forming
 * and using pointers into the gp cap-table region -- and `result` is a volatile
 * global. Globals x,y,result are all .bss. Single-TU. Checksum folds result and
 * the (post-swap) x,y; a native host folds the identical value. */
typedef unsigned long ulong;
typedef unsigned char bool;

static volatile int result;
static ulong px, py;

static bool divides(ulong n, ulong m) { return (m % n == 0); }
static bool even(ulong n) { return divides(2, n); }
static bool prime(ulong n) {
  ulong i;
  if (even(n)) return (n == 2);
  for (i = 3; i * i <= n; i += 2)
    if (divides(i, n)) return 0;
  return (n > 1);
}
static void swap(ulong *a, ulong *b) { ulong t = *a; *a = *b; *b = t; }

static unsigned prime_compute(void) {
  px = 21649UL;
  py = 513239UL;
  swap(&px, &py);
  result = (!(prime(px) && prime(py)));
  unsigned vals[3] = { (unsigned)result, (unsigned)px, (unsigned)py };
  unsigned h = 2166136261u; /* FNV-1a, LE bytes */
  for (int t = 0; t < 3; t++) {
    unsigned v = vals[t];
    for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
  }
  return h;
}
#endif
