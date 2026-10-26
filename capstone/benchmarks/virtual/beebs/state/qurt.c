/* State check for a BEEBS program whose verify_benchmark returns -1 (no verification upstream).
 * Appended to the program's own translation unit by build.py and called after the bracket; the value
 * is compared against the native build. It reads the program's output state, or re-calls a pure
 * function of it with the arguments benchmark() used when benchmark() discards the result. */
#include "bench-harness.h"
unsigned long long beebs_state_check(int r) {
  /* The globals hold the last of benchmark()'s three solves. */
  unsigned long long h = bench_fnv(BENCH_FNV_INIT, x1, sizeof x1);
  h = bench_fnv(h, x2, sizeof x2);
  h = bench_fnv(h, &flag, sizeof flag);
  int v = result;
  h = bench_fnv(h, &v, sizeof v);
  return bench_fnv(h, &r, sizeof r);
}
