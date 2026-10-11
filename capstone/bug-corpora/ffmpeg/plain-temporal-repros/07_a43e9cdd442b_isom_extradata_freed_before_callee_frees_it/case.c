#include "corpus.h"

/* libavformat/isom.c, ff_mp4_read_dec_config_descr, fix a43e9cdd442b.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(7) {
  /* Case 7 -- ff_mp4_read_dec_config_descr's extradata, fix a43e9cdd442b. The
   * freed object is st->codecpar->extradata; what is left holding its address is
   * the FIELD itself, which av_free does not clear.
   *
   * At the fix's parent:
   *
   *     av_free(st->codecpar->extradata);
   *     if ((ret = ff_get_extradata(fc, st->codecpar, pb, len)) < 0)
   *
   * and the fix deletes the caller's free, because ff_get_extradata releases the
   * old buffer itself -- so the field was being released twice.
   *
   * The reduction READS through the stale field where upstream FREES through it:
   * a real second free aborts in glibc, which is rc=134 and not a verdict, and
   * the dangling field is the same defect either way. */
  const unsigned long n = 48;
  unsigned char *extradata = malloc((size_t)n);
  CHECK(extradata, 871);
  memset(extradata, 0x11, (size_t)n);

  volatile unsigned char *field = extradata;      /* st->codecpar->extradata */
  if (!fixed) {
    free(extradata);                              /* the caller's av_free */
    o->freed = 1;
  } else {
    /* The fix: the caller does not free, so the field is still live when
     * ff_get_extradata is reached. The callee's own release is the ender. */
    o->freed = 1;
  }

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 872);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = read_probe(field);                /* what ff_get_extradata would release */
  o->aliased = o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "av_free left st->codecpar->extradata pointing at the released buffer, so "
                   "ff_get_extradata releases the same allocation a second time";
  o->fixed_text = "the fix removes the caller's free, leaving ff_get_extradata the single owner of "
                  "that step";
  free(fresh);
  if (fixed)
    free(extradata);
}
