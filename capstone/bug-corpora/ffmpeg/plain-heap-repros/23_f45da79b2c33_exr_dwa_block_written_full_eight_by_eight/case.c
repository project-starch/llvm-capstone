#include "corpus.h"

/* libavcodec/exr.c, dwa_uncompress's block loop, fix f45da79b2c33. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(23) {
  /* Case 23 -- dwa_uncompress's block loop, fix f45da79b2c33. PLAIN HEAP: the
   * loop steps x and y in 8s and writes a FULL 8x8 block each time.
   *
   * At the fix's parent:
   *
   *     for (int yy = 0; yy < 8; yy++) {
   *         for (int xx = 0; xx < 8; xx++) {
   *             const int idx = xx + yy * 8;
   *
   * and the fix clamps both to what remains of the tile:
   *
   *     int bw = FFMIN(8, td->xsize - x);
   *     int bh = FFMIN(8, td->ysize - y);
   *     ...
   *     for (int yy = 0; yy < bh; yy++) {
   *         for (int xx = 0; xx < bw; xx++) {
   *
   * With xsize or ysize not a multiple of 8 the last block writes past the row
   * and, on the last row, past the buffer.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long xsize = 4, ysize = 4;       /* NOT multiples of 8 */
  const unsigned long bytes = xsize * ysize;      /* 16 B: a size class */
  CHECK(xsize % 8 != 0, 1311);                    /* the premise, asserted */

  unsigned char *uncompressed = calloc((size_t)bytes, 1);
  CHECK(uncompressed, 1312);

  const unsigned long bw = fixed ? (8 < xsize ? 8 : xsize) : 8;
  const unsigned long bh = fixed ? (8 < ysize ? 8 : ysize) : 8;
  long touched = 0;
  int crossed = 0;
  for (unsigned long yy = 0; yy < bh && !crossed; yy++) {
    for (unsigned long xx = 0; xx < bw; xx++) {
      const unsigned long at = yy * xsize + xx;
      if (at >= bytes) {
        touched = (long)bytes;
        crossed = 1;
        write_probe_u8(uncompressed + bytes, 0xA5);   /* the labelled crossing */
        break;
      }
    }
  }

  o->cap = bytes;
  o->touched = touched;
  o->crossed = crossed;
  o->extent = (long)(8 * xsize + 8) - (long)bytes;
  o->damage = crossed;
  o->defect_text = "the DWA block loop wrote a full 8x8 block regardless of how much of it lay "
                   "inside the tile, so a tile whose dimensions are not multiples of 8 is written "
                   "past";
  o->fixed_text = "the fix clamps the block to FFMIN(8, xsize - x) by FFMIN(8, ysize - y)";
  free(uncompressed);
}
