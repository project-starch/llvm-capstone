#include "corpus.h"

/* libavcodec/exr.c, pxr24_uncompress's EXR_UINT branch, fix 162f75b5e679. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(24) {
  /* Case 24 -- pxr24_uncompress's UINT branch, fix 162f75b5e679. PLAIN HEAP: the
   * four byte planes are laid out and the cursor advanced by s->xdelta, the full
   * DATA WINDOW width, while td->tmp is sized by td->xsize, the TILE width.
   *
   * At the fix's parent:
   *
   *     ptr[1] = ptr[0] + s->xdelta;
   *     ptr[2] = ptr[1] + s->xdelta;
   *     ptr[3] = ptr[2] + s->xdelta;
   *     in     = ptr[3] + s->xdelta;
   *     for (j = 0; j < s->xdelta; ++j) {
   *
   * and the fix uses the tile's width, as the sibling FLOAT and HALF branches
   * already did:
   *
   *     ptr[1] = ptr[0] + td->xsize;
   *     ...
   *     for (j = 0; j < td->xsize; ++j) {
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long xsize = 4;                  /* the TILE width, which sized the buffer */
  const unsigned long xdelta = 16;                /* the DATA WINDOW width */
  const unsigned long bytes = 4 * xsize;          /* 16 B: a size class, four byte planes */
  CHECK(xdelta > xsize, 1321);                    /* the premise, asserted */

  unsigned char *tmp = calloc((size_t)bytes, 1);
  CHECK(tmp, 1322);

  const unsigned long step = fixed ? xsize : xdelta;
  /* The four plane cursors, and the loop that walks them. */
  long touched = 0;
  int crossed = 0;
  for (unsigned long j = 0; j < step && !crossed; j++) {
    for (unsigned long p = 0; p < 4; p++) {
      const unsigned long at = p * step + j;
      if (at >= bytes) {
        touched = (long)bytes;
        crossed = 1;
        (void)read_probe_u8(tmp + bytes);         /* the labelled crossing */
        break;
      }
    }
  }

  o->cap = bytes;
  o->touched = touched;
  o->crossed = crossed;
  o->extent = (long)(4 * (xdelta - xsize));
  o->damage = crossed;
  o->defect_text = "the UINT branch laid its four byte planes out by s->xdelta, the data-window "
                   "width, while the buffer was sized by td->xsize, the tile width";
  o->fixed_text = "the fix uses td->xsize, as the sibling FLOAT and HALF branches already did";
  free(tmp);
}
