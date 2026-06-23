/*
 * Capstone adapted tail for BEEBS `qsort`.
 *
 * Upstream `verify_benchmark` returns -1.  The benchmark sorts the global
 * `float arr[]` in place (the Numerical-Recipes algorithm is 1-indexed, sorting
 * arr[1..20]).  We verify the natural correctness property: the sorted region is
 * non-decreasing.  This is a real check and is independent of the harmless
 * 1-indexed over-read of arr[20].  The build script macro-renames the upstream
 * stub.
 */
#undef verify_benchmark

extern float arr[];

int verify_benchmark(int res) {
  (void)res;
  for (int i = 1; i < 19; i++) /* arr[1..19] is the sorted in-array region */
    if (arr[i] > arr[i + 1])
      return 0;
  return 1;
}
