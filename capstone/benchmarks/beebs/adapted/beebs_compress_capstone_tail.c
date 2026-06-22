/*
 * Capstone adapted tail for BEEBS `compress`.
 *
 * Upstream `verify_benchmark` returns -1 ("benchmark does not support
 * verification"), which the shared simple domain would treat as a trivially
 * CORRECT marker.  This BEEBS `compress` variant never calls `output()` (so
 * `comp_text_buffer`/`bytes_out` stay empty); the algorithm's real work
 * product is the LZW bookkeeping (`in_count`, `out_count`, `free_ent`) and the
 * hash tables (`htab`, `codetab`).  We replace `verify_benchmark` with an
 * FNV-1a hash over that end state and compare against a host reference run
 * (native LP64 build of the same source).  All values are deterministic
 * integer computations, identical on x86-64 and the Capstone target, so a
 * mismatch indicates a real capability-mode miscompile.
 *
 * The build script renames the upstream `initialise_benchmark` /
 * `verify_benchmark` via object-like macros; we undo those here and provide
 * the real definitions.
 */
#undef initialise_benchmark
#undef verify_benchmark

#define COMPRESS_HSIZE 400  /* HSIZE in libcompress.c */

/* Host reference (cc -O0, native LP64 build of libcompress.c):
 *   in_count=50 out_count=49 bytes_out=3 free_ent=306
 *   FNV-1a over {in_count,out_count,bytes_out,free_ent} + htab[] + codetab[] */
#define COMPRESS_EXPECTED_FNV 0xdd578ba2bbb979f4UL

extern long int in_count;
extern long int out_count;
extern long int bytes_out;
extern long int free_ent;
extern long int htab[COMPRESS_HSIZE];          /* count_int == long int */
extern unsigned short codetab[COMPRESS_HSIZE];

void initialise_benchmark(void) {}

int verify_benchmark(int res) {
  (void)res;
  unsigned long h = 1469598103934665603UL; /* FNV-1a offset basis */
#define COMPRESS_MIX(b)                                                        \
  do {                                                                         \
    h ^= (unsigned char)(b);                                                   \
    h *= 1099511628211UL; /* FNV-1a prime */                                   \
  } while (0)

  unsigned long scal[4] = {(unsigned long)in_count, (unsigned long)out_count,
                           (unsigned long)bytes_out, (unsigned long)free_ent};
  for (int s = 0; s < 4; s++)
    for (int k = 0; k < 8; k++)
      COMPRESS_MIX(scal[s] >> (8 * k));

  for (int i = 0; i < COMPRESS_HSIZE; i++) {
    unsigned long v = (unsigned long)htab[i];
    for (int k = 0; k < 8; k++)
      COMPRESS_MIX(v >> (8 * k));
  }
  for (int i = 0; i < COMPRESS_HSIZE; i++) {
    unsigned v = codetab[i];
    COMPRESS_MIX(v);
    COMPRESS_MIX(v >> 8);
  }
#undef COMPRESS_MIX

  return (h == COMPRESS_EXPECTED_FNV) ? 1 : 0;
}
