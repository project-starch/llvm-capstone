/*
 * Capstone adapted oracle for RV8 `miniz`.
 *
 * rv8-bench's miniz compresses then decompresses an 8 MB buffer and checks the
 * round-trip. The build reduces miniz to its core zlib-style compress/uncompress
 * (`-DMINIZ_NO_STDIO -DMINIZ_NO_ARCHIVE_APIS -DMINIZ_NO_TIME`, dropping the
 * FILE/zip/time code) and this tail drives a small fixed buffer:
 *   compress(src) -> cmp, then uncompress(cmp) -> unc, and check
 *   unc == src (round-trip) and cmp_len < src_len (real compression happened).
 * Self-contained oracle. miniz allocates a large (~200 KB) tdefl_compressor via
 * malloc internally, so the build enlarges the bump arena (-DRV8_HEAP_SIZE).
 * The src/cmp/unc buffers are small static arrays (no per-buffer malloc).
 */
#include "rv8_capstone_preamble.h"
#include <stddef.h>

typedef unsigned long mz_ulong;

extern int mz_compress(unsigned char *dst, mz_ulong *dlen,
                       const unsigned char *src, mz_ulong slen);
extern int mz_uncompress(unsigned char *dst, mz_ulong *dlen,
                         const unsigned char *src, mz_ulong slen);
extern void rv8_arena_init(void);

#ifndef RV8_MINIZ_N
#define RV8_MINIZ_N 4096
#endif

static unsigned char src_buf[RV8_MINIZ_N];
static unsigned char cmp_buf[RV8_MINIZ_N * 2]; /* > compressBound(N) */
static unsigned char unc_buf[RV8_MINIZ_N];

void initialise_benchmark(void) { rv8_arena_init(); }

int benchmark(void) {
  /* Mildly compressible, deterministic fill (low entropy -> deflate finds
     matches, so cmp_len < N exercises real LZ + Huffman). */
  for (size_t i = 0; i < RV8_MINIZ_N; i++)
    src_buf[i] = (unsigned char)((i * 7 + (i >> 3)) & 0x3f);

  mz_ulong cmp_len = sizeof(cmp_buf);
  if (mz_compress(cmp_buf, &cmp_len, src_buf, RV8_MINIZ_N) != 0) /* 0 == MZ_OK */
    return 0;
  if (cmp_len >= RV8_MINIZ_N) /* sanity: it actually compressed */
    return 0;

  mz_ulong unc_len = RV8_MINIZ_N;
  if (mz_uncompress(unc_buf, &unc_len, cmp_buf, cmp_len) != 0)
    return 0;
  if (unc_len != RV8_MINIZ_N)
    return 0;

  for (size_t i = 0; i < RV8_MINIZ_N; i++)
    if (unc_buf[i] != src_buf[i])
      return 0;

  return 1;
}

int verify_benchmark(int result) { return result == 1; }
