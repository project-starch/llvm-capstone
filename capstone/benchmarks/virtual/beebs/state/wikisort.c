/* State check for a BEEBS program whose verify_benchmark returns -1 (no verification upstream).
 * Appended to the program's own translation unit by build.py and called after the bracket; the value
 * is compared against the native build. It reads the program's output state, or re-calls a pure
 * function of it with the arguments benchmark() used when benchmark() discards the result. */
#include "bench-harness.h"
unsigned long long beebs_state_check(int result) {
  unsigned long long h = BENCH_FNV_INIT;
  for (long k = 0; k < max_size; k++) {
    long v[2] = {array1[k].value, array1[k].index};
    h = bench_fnv(h, v, sizeof v);
  }
  return bench_fnv(h, &result, sizeof result);
}
