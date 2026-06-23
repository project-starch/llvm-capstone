/*
 * Capstone adapted tail for BEEBS `newlib-mod`.
 *
 * Upstream `verify_benchmark` returns -1, and `benchmark()` overwrites a single
 * `volatile float result` across five `__ieee754_fmodf` calls (only the last
 * survives).  We rename both stubs: the adapted `benchmark()` captures all five
 * results, and `verify_benchmark` compares their bit patterns (exact) against a
 * native host reference.  `__ieee754_fmodf` is exact (shift-and-subtract on the
 * integer mantissa), so soft-float target output is bit-identical to the host
 * reference computed from the same source (gcc -O0 -ffp-contract=off).
 * The build script macro-renames the upstream stubs.
 */
#undef benchmark
#undef verify_benchmark

extern float __ieee754_fmodf(float, float);

static volatile float results[5];

int benchmark(void) {
  results[0] = __ieee754_fmodf(2.2353, 1234.5);
  results[1] = __ieee754_fmodf(3.2515, 2345.6);
  results[2] = __ieee754_fmodf(4.9346, 3456.7);
  results[3] = __ieee754_fmodf(5.2342, 4567.8);
  results[4] = __ieee754_fmodf(6.2352, 5678.9);
  return 0;
}

int verify_benchmark(int res) {
  (void)res;
  /* Host reference: the five __ieee754_fmodf pairs, gcc -O0 -ffp-contract=off. */
  static const unsigned expect[5] = {0x400f0f28U, 0x40501893U, 0x409de83eU,
                                     0x40a77e91U, 0x40c786c2U};
  union {
    float f;
    unsigned u;
  } b;
  for (int i = 0; i < 5; i++) {
    b.f = results[i];
    if (b.u != expect[i])
      return 0;
  }
  return 1;
}
