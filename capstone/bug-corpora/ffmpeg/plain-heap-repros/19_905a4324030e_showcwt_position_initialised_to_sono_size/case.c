#include "corpus.h"

/* libavfilter/avf_showcwt.c, the DU/RL position initialisation, fix 905a4324030e. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(19) {
  /* Case 19 -- showcwt's DU/RL position init, fix 905a4324030e. PLAIN HEAP: the
   * cursor was set to the row COUNT, so the first write goes one row past.
   *
   * At the fix's parent:
   *
   *     case DIRECTION_RL:
   *     case DIRECTION_DU:
   *         s->pos = s->sono_size;
   *         break;
   *
   * and the fix:
   *
   *         s->pos = FFMAX(s->sono_size - 1, 0);
   *
   * Valid rows are 0..sono_size-1, and with bar_ratio 0 sono_size == h.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long h = 4, linesize = 8;
  const unsigned long bytes = h * linesize;       /* 32 B: a size class */
  unsigned char *plane = calloc((size_t)bytes, 1);
  CHECK(plane, 1091);

  const unsigned long sono_size = h;              /* bar_ratio == 0 */
  const unsigned long pos = fixed ? (sono_size ? sono_size - 1 : 0) : sono_size;
  const unsigned long at = pos * linesize;

  o->cap = bytes;
  o->touched = (long)at;
  o->crossed = at >= bytes;
  o->extent = (long)linesize;
  o->damage = o->crossed;
  if (o->crossed)
    write_probe_u8(plane + at, 0xA5);             /* the labelled crossing */

  o->defect_text = "the row cursor was initialised to the row COUNT, so the first write lands one "
                   "whole row past the plane";
  o->fixed_text = "the fix initialises it to FFMAX(sono_size - 1, 0), the last valid row";
  free(plane);
}
