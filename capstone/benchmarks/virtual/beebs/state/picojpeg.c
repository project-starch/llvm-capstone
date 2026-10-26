/* State check for a BEEBS program whose verify_benchmark returns -1 (no verification upstream).
 * Appended to the program's own translation unit by build.py and called after the bracket; the value
 * is compared against the native build. It reads the program's output state, or re-calls a pure
 * function of it with the arguments benchmark() used when benchmark() discards the result. */
#include "bench-harness.h"
extern unsigned jpeg_off;
unsigned long long beebs_state_check(int result) {
  /* Appended to libpicojpeg.c, which owns the decoder state. The decoder leaves the last MCU and its coefficients in these buffers. */
  unsigned long long h = bench_fnv(BENCH_FNV_INIT, gMCUBufR, sizeof gMCUBufR);
  h = bench_fnv(h, gMCUBufG, sizeof gMCUBufG);
  h = bench_fnv(h, gMCUBufB, sizeof gMCUBufB);
  h = bench_fnv(h, gCoeffBuf, sizeof gCoeffBuf);
  h = bench_fnv(h, &jpeg_off, sizeof jpeg_off);
  return bench_fnv(h, &result, sizeof result);
}
