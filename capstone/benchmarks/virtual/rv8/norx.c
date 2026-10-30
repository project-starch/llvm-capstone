/* RV8 norx: upstream main() with DATA_SIZE = RV8_N bytes. As upstream, the tag is zeroed before
 * decryption (upstream cf_norx32_decrypt clears the plaintext on a tag MATCH), so the round trip
 * memcmp is 0. Deviation: pt1 is zeroed after malloc (see aes.c). */
#include RV8_UPSTREAM
#include "rv8-common.h"

static int cmp;
static unsigned long long ct_hash;

static void unit(void) {
  static const uint8_t key[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};
  static const uint8_t nonce[8] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};
  const size_t DATA_SIZE = RV8_N;
  uint8_t tag[16];
  uint8_t *pt1 = malloc(DATA_SIZE);
  uint8_t *ct = malloc(DATA_SIZE);
  uint8_t *pt2 = malloc(DATA_SIZE);
  memset(pt1, 0, DATA_SIZE);
  char c = 0x01;
  for (size_t j = 0; j < DATA_SIZE; j += sizeof(int))
    pt1[j] = (c ^= c * 7);
  bzero(tag, sizeof tag);
  cf_norx32_encrypt(key, nonce, NULL, 0, pt1, DATA_SIZE, NULL, 0, ct, tag);
  bzero(tag, sizeof tag);
  cf_norx32_decrypt(key, nonce, NULL, 0, ct, DATA_SIZE, NULL, 0, tag, pt2);
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

int main(int argc, char **argv) { return bench_run("rv8/norx", argc, argv, unit, post); }
