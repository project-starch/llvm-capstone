/*
 * Capstone adapted tail for BEEBS `newlib-log`.
 *
 * Upstream `verify_benchmark` returns -1, and `benchmark()` overwrites a single
 * `volatile float result` across five `__ieee754_logf` calls (only the last
 * survives).  We rename both stubs: the adapted `benchmark()` captures all five
 * results, and `verify_benchmark` compares their bit patterns (exact) against a
 * native host reference.  `__ieee754_logf` is a deterministic IEEE
 * single-precision routine (integer bit-manipulation + non-contracted float
 * arithmetic), so soft-float target output is bit-identical to the host
 * reference computed from the same source (gcc -O0 -ffp-contract=off).
 * The build script macro-renames the upstream stubs.
 */
#undef benchmark
#undef verify_benchmark

extern float __ieee754_logf(float);

static volatile float results[5];

int benchmark(void) {
  results[0] = __ieee754_logf(2);
  results[1] = __ieee754_logf(3);
  results[2] = __ieee754_logf(4);
  results[3] = __ieee754_logf(5);
  results[4] = __ieee754_logf(6);
  return 0;
}

int verify_benchmark(int res) {
  (void)res;
  /* Host reference: __ieee754_logf(2..6), gcc -O0 -ffp-contract=off. */
  static const unsigned expect[5] = {0x3f317218U, 0x3f8c9f54U, 0x3fb17218U,
                                     0x3fce0210U, 0x3fe55860U};
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
