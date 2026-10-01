/* Minimal libm stubs so FTS3/FTS5 (which reference log() etc. for ranking) LINK
 * in the freestanding Capstone domain. The CONTROL arm drives UAF code paths;
 * ranking VALUES are irrelevant to reachability, so cheap approximations are fine.
 * Only compiled into FTS builds (DOMAIN_EXTRA_SRC). */

/* log via range-reduction + series; good enough that rank ordering is finite,
 * never NaN/inf, so control flow through the ranker is well-defined. */
double log(double x) {
  if (x <= 0.0) return -1e300;           /* avoid inf/NaN */
  /* reduce x to [1,2): x = m * 2^e */
  int e = 0;
  while (x >= 2.0) { x *= 0.5; e++; }
  while (x < 1.0)  { x *= 2.0; e--; }
  /* ln(m) via atanh series, m in [1,2): t=(m-1)/(m+1) */
  double t = (x - 1.0) / (x + 1.0), t2 = t * t, sum = 0.0, term = t;
  for (int k = 1; k <= 15; k += 2) { sum += term / (double)k; term *= t2; }
  return 2.0 * sum + (double)e * 0.6931471805599453; /* + e*ln2 */
}
double log10(double x) { return log(x) * 0.4342944819032518; }
double log2(double x)  { return log(x) * 1.4426950408889634; }

double fabs(double x) { return x < 0.0 ? -x : x; }
double floor(double x){ double n=(double)(long long)x; return (n>x)?n-1.0:n; }
double ceil(double x) { double n=(double)(long long)x; return (n<x)?n+1.0:n; }

double sqrt(double x) {
  if (x <= 0.0) return 0.0;
  double g = x > 1.0 ? x : 1.0;
  for (int i = 0; i < 40; i++) g = 0.5 * (g + x / g);
  return g;
}
double pow(double b, double e) {
  /* integer-ish exponents are all FTS needs; general case via exp(e*log(b)) */
  if (b <= 0.0) return 0.0;
  return /*exp*/ ({ double y = e * log(b), s = 1.0, term = 1.0; for (int k=1;k<=20;k++){ term *= y/(double)k; s += term; } s; });
}
double exp(double y) { double s=1.0, term=1.0; for(int k=1;k<=25;k++){ term*=y/(double)k; s+=term; } return s; }


/* Generic qsort for FTS3 (freestanding build has no libc qsort). Insertion sort:
 * stable, correct, non-recursive; FTS arrays here are tiny so O(n^2) is fine. The
 * CONTROL arm only needs correct ordering so the query reaches the UAF path. */
void qsort(void *base, __SIZE_TYPE__ nmemb, __SIZE_TYPE__ size,
           int (*compar)(const void *, const void *)) {
  char *a = (char *)base;
  __SIZE_TYPE__ ps = sizeof(void *);
  /* Capstone: elements may hold capabilities (e.g. Fts3HashElem*). A byte-wise
   * swap copies the 16-byte capability as raw integer bytes, which CLEARS its tag,
   * so the next capability load of that element faults (cause 24, seen in
   * fts3CompareElemByTerm during term flush). When elements are pointer-sized and
   * aligned, swap in void* units (ldc/sdc preserve the tag); fall back to byte-wise
   * only for sub-pointer or misaligned data. */
  int capok = (size % ps == 0) && (((unsigned long)base % ps) == 0);
  for (__SIZE_TYPE__ i = 1; i < nmemb; i++) {
    for (__SIZE_TYPE__ j = i; j > 0 &&
         compar(a + (j - 1) * size, a + j * size) > 0; j--) {
      char *p = a + (j - 1) * size, *q = a + j * size;
      if (capok) {
        void **pp = (void **)p, **qq = (void **)q;
        for (__SIZE_TYPE__ k = 0; k < size / ps; k++) { void *t = pp[k]; pp[k] = qq[k]; qq[k] = t; }
      } else {
        for (__SIZE_TYPE__ k = 0; k < size; k++) { char tmp = p[k]; p[k] = q[k]; q[k] = tmp; }
      }
    }
  }
}
