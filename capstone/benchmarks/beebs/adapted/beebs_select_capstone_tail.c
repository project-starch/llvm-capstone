/*
 * Capstone adapted tail for BEEBS `select`.
 *
 * Upstream `verify_benchmark` returns -1 and `benchmark()` discards select()'s
 * return. We capture it and compare against a host reference. The build script
 * also widens `arr` to [21] so the 1-indexed Numerical-Recipes access (arr[1..n],
 * n=20) is in-bounds and deterministic. select() uses only float comparisons and
 * swaps, so the result is bit-identical across host and target. The build script
 * macro-renames the upstream benchmark/verify stubs.
 */
#undef benchmark
#undef verify_benchmark

extern float select(unsigned long k, unsigned long n);
extern int x, y; /* set to 10, 20 by the upstream initialise_benchmark */

static float select_result;

int benchmark(void) {
  select_result = select(x, y);
  return 0;
}

int verify_benchmark(int res) {
  (void)res;
  union {
    float f;
    unsigned u;
  } b;
  b.f = select_result;
  /* Host reference (cc -O0, arr[21]): select(10,20) = 10.3f = 0x4124cccd. */
  return (b.u == 0x4124cccdU) ? 1 : 0;
}
