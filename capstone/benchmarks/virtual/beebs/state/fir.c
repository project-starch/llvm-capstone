/* State check for a BEEBS program whose verify_benchmark returns -1 (no verification upstream).
 * Appended to the program's own translation unit by build.py and called after the bracket; the value
 * is compared against the native build. It reads the program's output state, or re-calls a pure
 * function of it with the arguments benchmark() used when benchmark() discards the result. */
#include "bench-harness.h"
static long fir_state_out[720];
unsigned long long beebs_state_check(int result) {
  /* benchmark() filters into a local array; filter again into a static one. */
  fir_filter_int(in_data, fir_state_out, 700, fir_int, 35, 285);
  unsigned long long h = bench_fnv(BENCH_FNV_INIT, fir_state_out, sizeof fir_state_out);
  return bench_fnv(h, &result, sizeof result);
}
