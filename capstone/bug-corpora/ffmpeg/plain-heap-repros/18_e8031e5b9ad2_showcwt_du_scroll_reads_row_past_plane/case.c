#include "corpus.h"

/* libavfilter/avf_showcwt.c, the DU-direction scroll, fix e8031e5b9ad2. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(18) {
  /* Case 18 -- showcwt's DU scroll, fix e8031e5b9ad2. PLAIN HEAP: each row is
   * copied from the row BELOW it, and the last iteration sources the row after
   * the final one.
   *
   * At the fix's parent:
   *
   *     for (int y = 0; y < s->sono_size; y++) {
   *         uint8_t *dst = s->outpicref->data[p] + y * linesize;
   *         memmove(dst, dst + linesize, s->w);
   *     }
   *
   * and the fix stops one row earlier:
   *
   *     for (int y = 0; y < s->sono_size - 1; y++) {
   *
   * With bar_ratio 0 the filter sets bar_size 0 and sono_size == h, so the final
   * source row is row h of a plane holding exactly h rows.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long h = 4, linesize = 8, w = 8;
  const unsigned long bytes = h * linesize;       /* 32 B: a size class */
  CHECK(bytes % 16 == 0, 1081);
  unsigned char *plane = calloc((size_t)bytes, 1);
  CHECK(plane, 1082);

  const unsigned long sono_size = h;              /* bar_ratio == 0 */
  const unsigned long bound = fixed ? sono_size - 1 : sono_size;
  long touched = 0;
  int crossed = 0;
  for (unsigned long y = 0; y < bound; y++) {
    const unsigned long src = (y + 1) * linesize;
    if (src >= bytes) {
      touched = (long)src;
      crossed = 1;
      (void)read_probe_u8(plane + src);           /* the labelled crossing */
      break;
    }
    memmove(plane + y * linesize, plane + src, (size_t)w);
  }

  o->cap = bytes;
  o->touched = touched;
  o->crossed = crossed;
  o->extent = (long)w;                            /* a whole row is read */
  o->damage = crossed;

  o->defect_text = "the DU scroll copies each row from the one below, so with sono_size == h the "
                   "last iteration sources row h of a plane holding exactly h rows";
  o->fixed_text = "the fix stops one row earlier, so the last source row is the final one";
  free(plane);
}
