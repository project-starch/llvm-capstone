/* State check for a BEEBS program whose verify_benchmark returns -1 (no verification upstream).
 * Appended to the program's own translation unit by build.py and called after the bracket; the value
 * is compared against the native build. It reads the program's output state, or re-calls a pure
 * function of it with the arguments benchmark() used when benchmark() discards the result. */
#include "bench-harness.h"
unsigned long long beebs_state_check(int result) {
  int v = binary_search(8); /* benchmark() discards it; binary_search reads only data[] */
  unsigned long long h = bench_fnv(BENCH_FNV_INIT, &v, sizeof v);
  return bench_fnv(h, &result, sizeof result);
}
