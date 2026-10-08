#include "corpus.h"

/* libavcodec/tdsc.c, tdsc_decode_tiles' raw-tile path, fix e9e6fb879835. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(15) {
  /* Case 15 -- tdsc's raw tile copy, fix e9e6fb879835. PLAIN HEAP: the buffer
   * is allocated for `tile_size` bytes read from the stream, and the copy reads
   * `3 * w * h` from a different pair of stream fields.
   *
   * The fix adds the missing relation:
   *
   *     if (3LL * w * h > tile_size)
   *         return AVERROR_INVALIDDATA;
   *
   * Nothing before it tied the two together.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long w = 8, h = 4;               /* the tile's declared geometry */
  const unsigned long tile_size = 48;             /* a size class, and smaller than 3*w*h == 96 */
  CHECK(3 * w * h > tile_size, 1051);             /* the premise, asserted */

  unsigned char *tilebuffer = calloc((size_t)tile_size, 1);
  CHECK(tilebuffer, 1052);

  const unsigned long want = 3 * w * h;
  if (fixed) {
    /* the fix refuses the tile outright */
    o->cap = tile_size;
    o->touched = 0;
    o->crossed = 0;
    o->extent = (long)want - (long)tile_size;
    o->damage = 0;
  } else {
    o->cap = tile_size;
    o->touched = (long)tile_size;                 /* the first byte past */
    o->crossed = 1;
    o->extent = (long)want - (long)tile_size;
    o->damage = 1;
    (void)read_probe_u8(tilebuffer + tile_size);  /* the labelled crossing */
  }

  o->defect_text = "the raw tile was copied by 3 * w * h while the buffer was allocated for the "
                   "stream's tile_size, with nothing relating the two";
  o->fixed_text = "the fix rejects a tile whose geometry needs more than tile_size bytes";
  free(tilebuffer);
}
