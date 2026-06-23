/*
 * Capstone adapted tail for BEEBS `qsort`.
 *
 * Upstream `verify_benchmark` returns -1.  The benchmark sorts the global
 * `float arr[]` in place (the Numerical-Recipes algorithm is 1-indexed,
 * sorting arr[1..20]). The build script widens arr to [21] so that region is
 * in-bounds. We verify both monotonicity and a byte hash of the full sorted
 * region against a native reference.
 */
#undef verify_benchmark

extern float arr[];

int verify_benchmark(int res) {
  (void)res;
  for (int i = 1; i < 20; i++)
    if (arr[i] > arr[i + 1])
      return 0;

  unsigned long h = 1469598103934665603UL;
  const unsigned char *p = (const unsigned char *)&arr[1];
  for (unsigned i = 0; i < sizeof(float) * 20; i++) {
    h ^= p[i];
    h *= 1099511628211UL;
  }
  return h == 0x9342ae3e06166c0aUL;
}
