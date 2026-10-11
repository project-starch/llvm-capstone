#include "corpus.h"

/* libavcodec/vlc.c, ff_vlc_init_multi_from_lengths, fix 2e04d35c69e6.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(12) {
  /* Case 12 -- ff_vlc_init_multi_from_lengths' scratch table, fix 2e04d35c69e6.
   * The freed object is the VLCcode table; it is released INSIDE a callee, and
   * the next call reads it.
   *
   * At the fix's parent:
   *
   *     ret = vlc_common_end(vlc, nb_bits, j, buf, flags, localbuf);
   *     if (ret < 0)
   *         goto fail;
   *     return vlc_multi_gen(multi->table, vlc, nb_elems, j, nb_bits, buf, logctx);
   *
   * vlc_common_end frees `buf` when it differs from the fallback it is handed.
   * The fix passes `buf` itself as that fallback, so the callee does not release
   * it, and frees it in the caller AFTER vlc_multi_gen. */
  const unsigned long n = 48;
  unsigned char *localbuf = malloc((size_t)n);    /* the caller's fallback */
  CHECK(localbuf, 921);
  unsigned char *buf = malloc((size_t)n);         /* the heap table actually used */
  CHECK(buf, 922);
  memset(buf, 0x11, (size_t)n);

  /* vlc_common_end's release: it frees its `buf` argument when it differs from
   * the fallback. The fix passes `buf` as the fallback, so they are equal. */
  unsigned char *fallback = fixed ? buf : localbuf;
  volatile unsigned char *stale = buf;
  if (buf != fallback) {
    free(buf);                                    /* the callee's av_free */
    o->freed = 1;
  } else {
    o->freed = 1;                                 /* the caller frees it after the use */
  }

  unsigned char *fresh = malloc((size_t)n);       /* tcache hands back the same chunk */
  CHECK(fresh, 923);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = read_probe(stale);                /* vlc_multi_gen's read of buf */
  o->aliased = o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "vlc_common_end freed the scratch table because it differed from the fallback it "
                   "was handed, and the next call read it";
  o->fixed_text = "the fix passes buf as its own fallback so the callee keeps it, and frees it in "
                  "the caller after the use";
  free(fresh);
  free(localbuf);
  if (fixed)
    free(buf);
}
