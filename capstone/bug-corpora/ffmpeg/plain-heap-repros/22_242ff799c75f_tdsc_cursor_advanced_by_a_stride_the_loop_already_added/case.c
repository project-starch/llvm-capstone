#include "corpus.h"

/* libavcodec/tdsc.c, tdsc_load_cursor's CUR_FMT_MONO paths, fix 242ff799c75f. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(22) {
  /* Case 22 -- tdsc_load_cursor's MONO paths, fix 242ff799c75f. PLAIN HEAP: the
   * inner loop already advances `dst` by a full stride per row, and a trailing
   * statement advanced it again.
   *
   * At the fix's parent, in each CUR_FMT_MONO case:
   *
   *             dst += ctx->cursor_stride - ctx->cursor_w * 4;
   *         }
   *
   * and the fix deletes both occurrences: the loop body alone advances a full
   * stride, stepping 32 pixels at a time at 4 bytes each, which is exactly
   * FFALIGN(cursor_w, 32) * 4 == cursor_stride.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long cursor_w = 1;
  const unsigned long cursor_h = 4;
  const unsigned long stride = 128;               /* FFALIGN(1, 32) * 4 */
  const unsigned long bytes = stride * cursor_h;  /* 512 B: a size class */
  const unsigned long extra = stride - cursor_w * 4;   /* the surplus per row */
  CHECK(extra > 0, 1301);                         /* the premise, asserted */

  unsigned char *cursor = calloc((size_t)bytes, 1);
  CHECK(cursor, 1302);

  unsigned long dst = 0;
  long touched = 0;
  int crossed = 0;
  for (unsigned long row = 0; row < cursor_h; row++) {
    /* The crossing is a row STARTING outside the buffer. A cursor that lands
     * exactly at `bytes` after the final row is correct, not a crossing, which
     * is why the test is on the row's start and not on the advanced cursor. */
    if (dst >= bytes) {
      touched = (long)bytes;
      crossed = 1;
      write_probe_u8(cursor + bytes, 0xA5);       /* the labelled crossing */
      break;
    }
    dst += stride;                                /* what the inner loop itself advances */
    if (!fixed)
      dst += extra;                               /* the trailing statement the fix deletes */
  }

  o->cap = bytes;
  o->touched = touched;
  o->crossed = crossed;
  o->extent = (long)((cursor_h - 1) * extra);
  o->damage = crossed;
  o->defect_text = "the inner loop already advanced dst by a full stride per row, and the trailing "
                   "statement advanced it again, so the later rows are written past the cursor "
                   "buffer";
  o->fixed_text = "the fix deletes the trailing advance, leaving the loop's own stride";
  free(cursor);
}
