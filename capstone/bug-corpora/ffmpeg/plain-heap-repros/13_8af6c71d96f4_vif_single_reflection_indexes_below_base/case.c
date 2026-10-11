#include "corpus.h"

/* libavfilter/vf_vif.c, vif_filter1d, fix 8af6c71d96f4. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(13) {
  /* Case 13 -- vif_filter1d's mirror, fix 8af6c71d96f4. PLAIN HEAP: the
   * hand-rolled mirror reflects ONCE, so an index far enough outside comes back
   * NEGATIVE.
   *
   * At the fix's parent:
   *
   *     ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);
   *
   * and the fix uses the library's mirror, which folds repeatedly:
   *
   *     ii = avpriv_mirror(ii, h - 1);
   *
   * With an index up to w - 1 + filt_w/2 and a small w, `2 * w - jj - 1` is
   * negative, so temp[jj] is read BELOW the buffer.
   *
   * THIS ONE LEAVES THE ALLOCATION, BELOW ITS BASE. */
  const unsigned long w = 4;                      /* the scale's width */
  const unsigned long filt_w = 17;                /* the filter width */
  const unsigned long elems = 4;                  /* 4 floats == 16 B: a size class */
  CHECK(w < filt_w / 2, 1031);                    /* the premise that makes it go negative */

  float *temp = calloc((size_t)elems, sizeof *temp);
  CHECK(temp, 1032);
  for (unsigned long i = 0; i < elems; i++)
    temp[i] = (float)i;

  const long jj_raw = (long)(w - 1 + filt_w / 2); /* the largest index the loop produces */
  long jj;
  if (fixed) {
    /* avpriv_mirror folds repeatedly until the index is in [0, w-1]. */
    jj = jj_raw;
    const long hi = (long)w - 1;
    while (jj < 0 || jj > hi)
      jj = jj < 0 ? -jj : 2 * hi - jj;
  } else {
    jj = jj_raw < 0 ? -jj_raw
                    : (jj_raw >= (long)w ? 2 * (long)w - jj_raw - 1 : jj_raw);
  }

  o->cap = elems;
  o->touched = jj;
  o->crossed = jj < 0 || jj >= (long)elems;
  o->extent = jj < 0 ? -jj : 1;
  o->damage = o->crossed;
  if (o->crossed)
    (void)read_probe(temp + jj);                  /* the labelled crossing, BELOW the base */

  o->defect_text = "the hand-rolled mirror reflects only once, so an index beyond w + filt_w/2 comes "
                   "back negative and temp[jj] reads below the buffer's base";
  o->fixed_text = "the fix uses avpriv_mirror, which folds repeatedly into [0, w-1]";
  free(temp);
}
