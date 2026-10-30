/* RV8 aes: upstream main() with DATA_SIZE = RV8_N bytes. Deviation: pt1 is zeroed after malloc --
 * upstream writes every fourth byte and relies on fresh pages for the rest, which a long-running
 * process reusing freed memory does not give (T1 keeps freed bytes, T2 scrubs them). */
#include RV8_UPSTREAM
#include "rv8-common.h"

static int cmp;
static unsigned long long ct_hash;

static void unit(void) {
  static const u8 key[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                             0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};
  const size_t DATA_SIZE = RV8_N;
  u8 *pt1 = malloc(DATA_SIZE);
  u8 *ct = malloc(DATA_SIZE);
  u8 *pt2 = malloc(DATA_SIZE);
  memset(pt1, 0, DATA_SIZE);
  char c = 0x01;
  for (size_t j = 0; j < DATA_SIZE; j += sizeof(int))
    pt1[j] = (c ^= c * 7);
  void *rk1 = aes_encrypt_init(key, 16);
  for (size_t j = 0; j < DATA_SIZE; j += 16)
    aes_encrypt((u32 *)rk1, pt1 + j, ct + j);
  aes_encrypt_deinit(rk1);
  void *rk2 = aes_decrypt_init(key, 16);
  for (size_t j = 0; j < DATA_SIZE; j += 16)
    aes_decrypt((u32 *)rk2, ct + j, pt2 + j);
  aes_decrypt_deinit(rk2);
  cmp = memcmp(pt1, pt2, DATA_SIZE);
  ct_hash = bench_fnv(bench_fnv(BENCH_FNV_INIT, ct, 16), ct + DATA_SIZE - 16, 16);
  free(pt1);
  free(ct);
  free(pt2);
}

static int post(unsigned long long *check) {
  *check = ct_hash ^ (unsigned long long)(unsigned)cmp;
  return cmp == 0;
}

int main(int argc, char **argv) { return bench_run("rv8/aes", argc, argv, unit, post); }
