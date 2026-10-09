#include "corpus.h"

/* VP9's intra-prediction row backup, fix 2563a33856 ("re-initialize internal
 * buffers on bpp change also"). The same assign() block as case 6, carved at
 * the bit depth of the moment (libavcodec/vp9.c of the fix's parent):
 *
 *     int bytesperpixel = s->bytesperpixel;                                   // :316
 *     p = av_malloc(s->sb_cols * (128 + 192 * bytesperpixel +
 *                                 sizeof(*s->lflvl) + 16 * sizeof(*s->above_mv_ctx)));  // :335-336
 *     assign(s->intra_pred_data[0],  uint8_t *,             64 * bytesperpixel);  // :339
 *
 * With frame threads, the next thread's update_size() returns early because
 * the frame size and pix_fmt it is handed already match (:320), while
 * vp9_decode_update_thread_context() frees the buffers only on a cols/rows
 * change (:4321-4323) but copies `s->bytesperpixel = ssrc->bytesperpixel`
 * (:4351). After an 8-to-10-bit switch that thread decodes at 2 bytes per
 * pixel into regions carved for 1, and the row backup (:4207-4209)
 *
 *     memcpy(s->intra_pred_data[0], f->data[0] + yoff + 63 * ls_y,
 *            8 * s->cols * bytesperpixel);
 *
 * writes twice the region: intra_pred_data[0] runs over [1]. The fix also
 * frees on a bpp change, so update_size() re-carves at the new depth. The
 * threading is not modelled; its effect, a carve at bpp 1 used at bpp 2, is.
 *
 * The chroma backup (:4210-4212) then refills [1] in place, so the spill is
 * overwritten -- and the next superblock row's luma intra prediction reads
 * intra_pred_data[0] at 2 bytes per pixel, half of it chroma. */
FFC_CASE(7) {
  const size_t sb_cols = 2, cols = 16; /* a 128-pixel-wide frame */
  const size_t carve_bpp = fixed ? 2 : 1, bpp = 2;
  const size_t lflvl = 64 + 2 * 2 * 8 * 4, mv = 16 * 2 * 4;
  const size_t bytes = sb_cols * (128 + 192 * carve_bpp + lflvl + mv);

  unsigned char *p = malloc(bytes);
  CHECK(p, 771);
  /* :339-355, the assign() order at the parent. */
  const struct { const char *name; size_t n; } list[] = {
      {"intra_pred_data[0]", 64 * carve_bpp}, {"intra_pred_data[1]", 64 * carve_bpp},
      {"intra_pred_data[2]", 64 * carve_bpp}, {"above_y_nnz_ctx", 16},
      {"above_mode_ctx", 16},                 {"above_mv_ctx", mv},
      {"above_uv_nnz_ctx[0]", 16},            {"above_uv_nnz_ctx[1]", 16},
      {"above_partition_ctx", 8},             {"above_skip_ctx", 8},
      {"above_txfm_ctx", 8},                  {"above_segpred_ctx", 8},
      {"above_intra_ctx", 8},                 {"above_comp_ctx", 8},
      {"above_ref_ctx", 8},                   {"above_filter_ctx", 8},
      {"lflvl", lflvl}};
  unsigned char *intra[3];
  size_t off = 0;
  for (size_t i = 0; i < sizeof list / sizeof *list; i++) {
    unsigned char *r = ffc_carve(p, off, sb_cols * list[i].n, list[i].name);
    if (i < 3)
      intra[i] = r;
    off += sb_cols * list[i].n;
  }
  CHECK(off == bytes, 772);

  const size_t region = sb_cols * 64 * carve_bpp;
  const size_t luma = 8 * cols * bpp;   /* :4209 */
  const size_t chroma = luma >> 1;      /* :4212, 4:2:0 */
  unsigned char row[256];
  CHECK(luma <= sizeof row, 773);
  memset(row, 0x77, sizeof row);        /* the luma row being backed up */
  size_t inside = luma < region ? luma : region;
  memcpy(intra[0], row, inside);
  if (luma > region) {
    ffc_note(o, p, bytes, intra[0], region, intra[0] + region, luma - region);
    write_probe_u8(intra[0] + region, row[region]);
    memcpy(intra[0] + region + 1, row + region + 1, luma - region - 1);
  } else {
    ffc_note(o, p, bytes, intra[0], region, intra[0] + luma - 1, 1);
  }
  /* The chroma backup into [1], then the next row's luma prediction. */
  memset(intra[1], 0x88, chroma);
  o->damage = intra[0][luma - 1] != 0x77;

  o->defect_text = "a thread at 10-bit backed up 256 luma bytes into an intra_pred_data[0] "
                   "carved for 8-bit, 128 bytes, running into intra_pred_data[1]";
  o->fixed_text = "the fix re-carves on a bit-depth change, so [0] is 256 bytes";
  free(p);
}
