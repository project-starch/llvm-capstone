/*
 * Capstone adapted tail for BEEBS `newlib-sqrt`.
 *
 * Upstream `verify_benchmark` already does a real check: exact float equality
 * of result[0..5] against a hardcoded `exp[]`.  `__ieee754_sqrtf` is
 * correctly-rounded (bit-by-bit), so the soft-float target output is
 * bit-identical to the embedded newlib values (cross-checked on the host).
 *
 * The only adaptation is Bug #9: the upstream `exp[]` is a *local* const array
 * in `verify_benchmark`, which the backend lowers to a `memcpy` from `.rodata`
 * into a stack array whose destination capability comes back untagged.  Marking
 * it `static const` lands it in `.rodata` (no stack copy, no `memcpy`) — the
 * same class of fix used by ludcmp / nettle-* / mergesort.  The build script
 * macro-renames the upstream stub.
 */
#undef verify_benchmark

extern volatile float result[6];

int verify_benchmark(int unused) {
  (void)unused;
  static const float exp[] = {1.41421353816986083984375f,
                              1.73205077648162841796875f,
                              2.2360680103302001953125f,
                              2.4494898319244384765625f,
                              2.6457512378692626953125f,
                              2.8284270763397216796875f};
  for (int i = 0; i < 6; i++)
    if (result[i] != exp[i])
      return 0;
  return 1;
}
