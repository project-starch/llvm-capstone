/*
 * Capstone adapted oracle for RV8 `sha512`.
 *
 * rv8-bench's sha512 is a self-contained SHA-512 (no malloc). main() hashes
 * 1,000,000 * 64 zero bytes and prints the digest. This tail drives the same
 * upstream sha512_init/update/final over a reduced, fixed input (RV8_SHA512_ROUNDS
 * * 64 zero bytes; perf count is irrelevant to correctness) and checks the 64-byte
 * digest against a host-derived reference for the same input -- a deterministic,
 * self-contained oracle.
 *
 * struct sha512_ctx_t is redeclared here to match the upstream definition (it is
 * file-local in sha512.c). `expected` is the digest of RV8_SHA512_ROUNDS*64 zero
 * bytes, computed with a native gcc build of the same source.
 */
#include "rv8_capstone_preamble.h"
#include <stdint.h>

struct sha512_ctx_t {
  uint64_t chain[8];
  uint8_t block[128];
  uint64_t nbytes;
};

extern void sha512_init(struct sha512_ctx_t *ctx);
extern void sha512_update(struct sha512_ctx_t *ctx, const unsigned char *data,
                          uint64_t bytes);
extern void sha512_final(struct sha512_ctx_t *ctx, uint8_t result[64]);
extern void rv8_arena_init(void);

#ifndef RV8_SHA512_ROUNDS
#define RV8_SHA512_ROUNDS 1000
#endif

/* SHA-512 of RV8_SHA512_ROUNDS*64 (= 64000) zero bytes; native gcc reference. */
static const uint8_t expected[64] = {
    0xcb, 0x64, 0xd4, 0x56, 0xcd, 0xc8, 0x29, 0xc1, 0x65, 0xf6, 0x76, 0x2c,
    0x6c, 0x56, 0x2c, 0x36, 0x9e, 0xeb, 0x5d, 0x8d, 0x7a, 0x30, 0x96, 0x43,
    0xdc, 0x78, 0xb9, 0xdd, 0x22, 0x77, 0x00, 0x96, 0x08, 0x2b, 0xaf, 0xd1,
    0xbc, 0x21, 0xcf, 0x5e, 0xe1, 0x12, 0xe0, 0x02, 0x9f, 0x0d, 0x36, 0x9c,
    0x90, 0xcd, 0x97, 0x4f, 0x97, 0x89, 0xbc, 0x3d, 0x6f, 0xc8, 0x3a, 0x0a,
    0x71, 0x2e, 0x8a, 0xe7};

void initialise_benchmark(void) { rv8_arena_init(); }

int benchmark(void) {
  struct sha512_ctx_t ctx;
  unsigned char buf[64];
  uint8_t output[64];

  for (int i = 0; i < 64; i++)
    buf[i] = 0;

  sha512_init(&ctx);
  for (size_t i = 0; i < RV8_SHA512_ROUNDS; i++)
    sha512_update(&ctx, buf, 64);
  sha512_final(&ctx, output);

  int ok = 1;
  for (int i = 0; i < 64; i++)
    if (output[i] != expected[i])
      ok = 0;
  return ok;
}

int verify_benchmark(int result) { return result == 1; }
