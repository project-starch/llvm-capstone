/* RV8 sha512: upstream main() with RV8_N updates of 64 zero bytes (upstream: 1000000). Upstream
 * prints the digest; the check is a hash of it, compared against the native build. */
#include RV8_UPSTREAM
#include "rv8-common.h"

static uint8_t output[64];

static void unit(void) {
  struct sha512_ctx_t sha512;
  uint8_t buf[64];
  memset(buf, 0, SHA512_OUTPUT_BYTES);
  sha512_init(&sha512);
  for (size_t i = 0; i < RV8_N; i++)
    sha512_update(&sha512, buf, SHA512_OUTPUT_BYTES);
  sha512_final(&sha512, output);
}

static int post(unsigned long long *check) {
  *check = bench_fnv(BENCH_FNV_INIT, output, sizeof output);
  /* Upstream has no expected digest; a digest of all zeros means sha512_final never ran. */
  for (size_t i = 0; i < sizeof output; i++)
    if (output[i])
      return 1;
  return 0;
}

int main(int argc, char **argv) { return bench_run("rv8/sha512", argc, argv, unit, post); }
