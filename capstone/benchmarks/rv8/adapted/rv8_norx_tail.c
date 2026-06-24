/*
 * Capstone adapted oracle for RV8 `norx`.
 *
 * rv8-bench's norx is the NORX32 AEAD cipher. main() encrypts a 32 MB malloc'd
 * buffer, then (mirroring its own code) zeroes the tag and decrypts, checking
 * the round-trip (memcmp==0). Note the upstream cf_norx32_decrypt has inverted
 * tag logic (it zeroes the plaintext on a tag *match*), so main deliberately
 * passes a zeroed tag to recover the plaintext -- we replicate that exactly.
 *
 * This tail uses small static buffers (no 32 MB malloc) filled with the same
 * pattern and a self-contained oracle: the decrypt round-trips the plaintext
 * (pt2 == pt1) AND the ciphertext actually differs from the plaintext (the
 * cipher transformed the data, ruling out a trivial identity pass).
 */
#include "rv8_capstone_preamble.h"
#include <stdint.h>
#include <stddef.h>

extern void cf_norx32_encrypt(const uint8_t *key, const uint8_t *nonce,
                              const uint8_t *header, size_t nheader,
                              const uint8_t *plaintext, size_t nbytes,
                              const uint8_t *trailer, size_t ntrailer,
                              uint8_t *ciphertext, uint8_t *tag);
extern int cf_norx32_decrypt(const uint8_t *key, const uint8_t *nonce,
                             const uint8_t *header, size_t nheader,
                             const uint8_t *ciphertext, size_t nbytes,
                             const uint8_t *trailer, size_t ntrailer,
                             const uint8_t *tag, uint8_t *plaintext);
extern void rv8_arena_init(void);

#ifndef RV8_NORX_N
#define RV8_NORX_N 1024
#endif

static uint8_t pt1[RV8_NORX_N];
static uint8_t ct[RV8_NORX_N];
static uint8_t pt2[RV8_NORX_N];

void initialise_benchmark(void) { rv8_arena_init(); }

int benchmark(void) {
  static const uint8_t key[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                  0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                                  0x0e, 0x0f};
  static const uint8_t nonce[8] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};
  uint8_t tag[16];

  /* Same sparse fill pattern as upstream main() (every sizeof(int)th byte). */
  char c = 0x01;
  for (size_t j = 0; j < RV8_NORX_N; j += sizeof(int))
    pt1[j] = (c ^= c * 7);

  for (int i = 0; i < 16; i++)
    tag[i] = 0;
  cf_norx32_encrypt(key, nonce, NULL, 0, pt1, RV8_NORX_N, NULL, 0, ct, tag);

  for (int i = 0; i < 16; i++)
    tag[i] = 0; /* mirror main: zeroed tag so decrypt keeps the plaintext */
  cf_norx32_decrypt(key, nonce, NULL, 0, ct, RV8_NORX_N, NULL, 0, tag, pt2);

  int roundtrip = 1;
  int transformed = 0;
  for (size_t i = 0; i < RV8_NORX_N; i++) {
    if (pt2[i] != pt1[i])
      roundtrip = 0;
    if (ct[i] != pt1[i])
      transformed = 1;
  }
  return (roundtrip && transformed) ? 1 : 0;
}

int verify_benchmark(int result) { return result == 1; }
