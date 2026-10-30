/* State check for a BEEBS program whose verify_benchmark returns -1 (no verification upstream).
 * Appended to the program's own translation unit by build.py and called after the bracket; the value
 * is compared against the native build. It reads the program's output state, or re-calls a pure
 * function of it with the arguments benchmark() used when benchmark() discards the result. */
#include "bench-harness.h"
unsigned long long beebs_state_check(int result) {
  double d[8] = {T, T1, T2, E1[0], E1[1], E1[2], E1[3], E1[4]};
  int i[3] = {J, K, L};
  unsigned long long h = bench_fnv(BENCH_FNV_INIT, d, sizeof d);
  h = bench_fnv(h, i, sizeof i);
  return bench_fnv(h, &result, sizeof result);
}
