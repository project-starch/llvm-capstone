#include "corpus.h"

/* svq1enc's scratch buffer, fix 043bcdcdb0 ("fix encoding of small widths").
 * ONE allocation (libavcodec/svq1enc.c:589 of the fix's parent)
 *
 *     s->scratchbuf = av_malloc(s->current_picture->linesize[0] * 16 * 2);
 *
 * that svq1_encode_plane() carves into a reconstruction area and the source
 * block being encoded:
 *
 *     uint8_t *src     = s->scratchbuf + stride * 16;                         // :255
 *     uint8_t *temp    = s->scratchbuf;                                       // :369
 *
 * temp holds the intra reconstruction in columns 0..15 and the inter
 * prediction at `temp + 16` (:430-433):
 *
 *     s->hdsp.put_pixels_tab[0][dxy](temp + 16, ref + ..., stride, 16);
 *
 * sixteen rows of sixteen bytes, the last at 15 * stride + 31. When the plane's
 * stride is below 32 that is past temp's 16 * stride bytes: the prediction's
 * last row lands on src's first, and encode_block() at :435 then encodes
 * against a corrupted source. The fix moves src to `stride * 32`, the
 * prediction to `temp + 16 * stride`, and allocates 16 * 3 rows.
 *
 * The stride is the CHROMA plane's (svq1 is 4:1:0, so an 8-pixel chroma row
 * with 16-byte alignment) while the allocation is sized by the luma linesize,
 * 32: the plane encoded is narrower than the buffer, and the overshoot stays
 * inside it. Columns 16..31 of a 16-byte stride also fold onto temp's next row
 * -- the 2-D overlap this tree does not count, since it never leaves temp. */
FFC_CASE(9) {
  const size_t luma_linesize = 32, stride = 16;
  const size_t rows = fixed ? 16 * 3 : 16 * 2;                    /* :589, and the fix */
  const size_t bytes = luma_linesize * rows;
  const size_t temp_len = fixed ? 32 * stride : 16 * stride;     /* up to src */
  const size_t pred_off = fixed ? 16 * stride : 16;              /* :430, and the fix */

  unsigned char *scratchbuf = malloc(bytes);
  CHECK(scratchbuf, 791);
  unsigned char *temp = ffc_carve(scratchbuf, 0, temp_len, "temp");
  unsigned char *src = ffc_carve(scratchbuf, temp_len, 16 * stride, "src");
  CHECK(temp_len + 16 * stride <= bytes, 792);
  for (size_t r = 0; r < 16; r++)
    memset(src + r * stride, 0x40 + (int)r, 16); /* the source block, row by row */

  /* put_pixels16 into temp + pred_off: 16 rows of 16 bytes at `stride`. */
  unsigned char *pred = temp + pred_off;
  size_t r = 0;
  for (; r < 16 && pred_off + r * stride + 16 <= temp_len; r++)
    memset(pred + r * stride, 0x99, 16);
  if (r < 16) {
    unsigned char *at = pred + r * stride;
    ffc_note(o, scratchbuf, bytes, temp, temp_len, at,
             (size_t)(pred + 15 * stride + 16 - at));
    write_probe_u8(at, 0x99);
    memset(at + 1, 0x99, 15);
  } else {
    ffc_note(o, scratchbuf, bytes, temp, temp_len, pred + 15 * stride + 15, 1);
  }
  o->damage = src[0] != 0x40; /* the block encode_block() then reads */

  o->defect_text = "the 16x16 inter prediction at temp + 16 with a 16-byte stride ended past "
                   "temp's 16 rows, overwriting src's first row";
  o->fixed_text = "the fix predicts at temp + 16 * stride inside a 32-row temp, and src starts after it";
  free(scratchbuf);
}
