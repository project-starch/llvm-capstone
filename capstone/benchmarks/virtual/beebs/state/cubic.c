/* State check for a BEEBS program whose verify_benchmark returns -1 (no verification upstream).
 * Appended to the program's own translation unit by build.py and called after the bracket; the value
 * is compared against the native build. It reads the program's output state, or re-calls a pure
 * function of it with the arguments benchmark() used when benchmark() discards the result. */
#include "bench-harness.h"
unsigned long long beebs_state_check(int result) {
  /* benchmark() writes its roots to a local array; solve its first four cubics again. */
  double out[12] = {0};
  int n[4];
  SolveCubic(1.0, -10.5, 32.0, -30.0, &n[0], out);
  SolveCubic(1.0, -4.5, 17.0, -30.0, &n[1], out + 3);
  SolveCubic(1.0, -3.5, 22.0, -31.0, &n[2], out + 6);
  SolveCubic(1.0, -13.7, 1.0, -35.0, &n[3], out + 9);
  unsigned long long h = bench_fnv(BENCH_FNV_INIT, n, sizeof n);
  h = bench_fnv(h, out, sizeof out);
  return bench_fnv(h, &result, sizeof result);
}
