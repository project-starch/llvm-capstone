/*
 * Capstone adapted oracle for RV8 `aes`.
 *
 * rv8-bench's aes is a standard Rijndael; main() encrypts then decrypts a large
 * malloc'd buffer and checks the round-trip (memcmp==0). This tail drives the
 * same upstream encrypt/decrypt API with a strong, self-contained oracle:
 *   1. the FIPS-197 AES-128 known-answer (key 000102..0F, plaintext 0011..FF ->
 *      ciphertext 69C4E0D8..C55A), and
 *   2. the encrypt/decrypt round-trip (decrypt(encrypt(pt)) == pt).
 * (1) catches a wrong cipher; (2) catches an asymmetric encrypt/decrypt bug.
 * No large buffers; only the small malloc'd round-key context (AES_PRIV_SIZE).
 */
#include "rv8_capstone_preamble.h"

typedef unsigned char u8;

extern void *aes_encrypt_init(const u8 *key, size_t len);
extern void aes_encrypt(void *ctx, const u8 *plain, u8 *crypt);
extern void aes_encrypt_deinit(void *ctx);
extern void *aes_decrypt_init(const u8 *key, size_t len);
extern void aes_decrypt(void *ctx, const u8 *crypt, u8 *plain);
extern void aes_decrypt_deinit(void *ctx);
extern void rv8_arena_init(void);

void initialise_benchmark(void) { rv8_arena_init(); }

int benchmark(void) {
  /* FIPS-197 AES-128 example (Appendix B / C.1). */
  static const u8 key[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                             0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  static const u8 pt[16] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                            0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  static const u8 expect_ct[16] = {0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04,
                                   0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4,
                                   0xc5, 0x5a};
  u8 ct[16];
  u8 pt2[16];

  void *rke = aes_encrypt_init(key, 16);
  if (!rke)
    return 0;
  aes_encrypt(rke, pt, ct);
  aes_encrypt_deinit(rke);

  void *rkd = aes_decrypt_init(key, 16);
  if (!rkd)
    return 0;
  aes_decrypt(rkd, ct, pt2);
  aes_decrypt_deinit(rkd);

  int ok = 1;
  for (int i = 0; i < 16; i++) {
    if (ct[i] != expect_ct[i])
      ok = 0; /* known-answer */
    if (pt2[i] != pt[i])
      ok = 0; /* round-trip */
  }
  return ok;
}

int verify_benchmark(int result) { return result == 1; }
