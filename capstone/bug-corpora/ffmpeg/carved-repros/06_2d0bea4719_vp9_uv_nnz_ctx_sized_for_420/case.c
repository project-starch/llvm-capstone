#include "corpus.h"

/* VP9's above-context block, fix 2d0bea4719 ("increase buffer sizes for
 * non-420 chroma subsamplings"). update_size() takes ONE av_malloc and carves
 * it with assign() (libavcodec/vp9.c:327-347 of the fix's parent):
 *
 *     #define assign(var, type, n) var = (type) p; p += s->sb_cols * (n) * sizeof(*var)
 *     p = av_malloc(s->sb_cols * (240 + sizeof(*s->lflvl) + 16 * sizeof(*s->above_mv_ctx)));
 *     ...
 *     assign(s->above_uv_nnz_ctx[0], uint8_t *,              8);
 *     assign(s->above_uv_nnz_ctx[1], uint8_t *,              8);
 *     assign(s->above_segpred_ctx,   uint8_t *,              8);
 *
 * Eight bytes per superblock column is the 4:2:0 size. The frame setup clears
 * them by the stream's subsampling (:3877-3878):
 *
 *     memset(s->above_uv_nnz_ctx[0], 0, s->sb_cols * 16 >> s->ss_h);
 *
 * so a 4:4:4 or 4:4:0 frame (ss_h = 0, accepted for profile 1 at :485-499)
 * writes 16 * sb_cols bytes into an 8 * sb_cols region: [0] runs over [1], and
 * [1] over above_segpred_ctx. The fix carves both at 16 and grows the block to
 * 320 + lflvl + mv bytes per column.
 *
 * NO OBSERVABLE DAMAGE, and recorded as such: the bytes are zeros, over
 * regions that are themselves cleared next. The crossing is real; the
 * corruption is not. This is the row where only a bound can tell. */
struct carve {
  const char *name;
  size_t n; /* bytes per superblock column */
};

FFC_CASE(6) {
  const size_t sb_cols = 2;             /* a 128-pixel-wide frame */
  const size_t lflvl = 64 + 2 * 2 * 8 * 4; /* sizeof(struct VP9Filter), :80-84 */
  const size_t mv = 16 * 2 * 4;         /* 16 * sizeof(VP56mv[2]) */
  const size_t uv = fixed ? 16 : 8;
  /* The carve order at the parent, and at the fix (which also widens
   * intra_pred_data[1..2] to 64 and moves the uv contexts after the mv one). */
  const struct carve parent[] = {
      {"intra_pred_data[0]", 64}, {"intra_pred_data[1]", 32}, {"intra_pred_data[2]", 32},
      {"above_y_nnz_ctx", 16},    {"above_mode_ctx", 16},     {"above_mv_ctx", mv},
      {"above_partition_ctx", 8}, {"above_skip_ctx", 8},      {"above_txfm_ctx", 8},
      {"above_uv_nnz_ctx[0]", 8}, {"above_uv_nnz_ctx[1]", 8}, {"above_segpred_ctx", 8},
      {"above_intra_ctx", 8},     {"above_comp_ctx", 8},      {"above_ref_ctx", 8},
      {"above_filter_ctx", 8},    {"lflvl", lflvl}};
  const struct carve fix[] = {
      {"intra_pred_data[0]", 64}, {"intra_pred_data[1]", 64}, {"intra_pred_data[2]", 64},
      {"above_y_nnz_ctx", 16},    {"above_mode_ctx", 16},     {"above_mv_ctx", mv},
      {"above_uv_nnz_ctx[0]", uv}, {"above_uv_nnz_ctx[1]", uv}, {"above_partition_ctx", 8},
      {"above_skip_ctx", 8},      {"above_txfm_ctx", 8},      {"above_segpred_ctx", 8},
      {"above_intra_ctx", 8},     {"above_comp_ctx", 8},      {"above_ref_ctx", 8},
      {"above_filter_ctx", 8},    {"lflvl", lflvl}};
  const struct carve *list = fixed ? fix : parent;
  const size_t count = fixed ? sizeof fix / sizeof *fix : sizeof parent / sizeof *parent;
  const size_t bytes = sb_cols * ((fixed ? 320 : 240) + lflvl + mv);

  unsigned char *p = malloc(bytes);
  CHECK(p, 761);
  unsigned char *uv_nnz[2] = {0, 0};
  size_t off = 0;
  for (size_t i = 0; i < count; i++) {
    unsigned char *r = ffc_carve(p, off, sb_cols * list[i].n, list[i].name);
    if (!strcmp(list[i].name, "above_uv_nnz_ctx[0]"))
      uv_nnz[0] = r;
    if (!strcmp(list[i].name, "above_uv_nnz_ctx[1]"))
      uv_nnz[1] = r;
    off += sb_cols * list[i].n;
  }
  CHECK(off == bytes && uv_nnz[0] && uv_nnz[1], 762); /* the carve fills the block */

  const int ss_h = 0;                   /* 4:4:4 */
  const size_t clear = sb_cols * 16 >> ss_h;
  const size_t region = sb_cols * uv;
  size_t inside = clear < region ? clear : region;
  memset(uv_nnz[0], 0, inside);
  if (clear > region) {
    ffc_note(o, p, bytes, uv_nnz[0], region, uv_nnz[0] + region, clear - region);
    write_probe_u8(uv_nnz[0] + region, 0);
  } else {
    ffc_note(o, p, bytes, uv_nnz[0], region, uv_nnz[0] + clear - 1, 1);
  }
  o->damage = 0; /* zeros, over a region the next statement clears anyway */

  o->defect_text = "the 4:4:4 clear wrote 16 bytes per superblock column into an 8-byte "
                   "above_uv_nnz_ctx[0], running into above_uv_nnz_ctx[1]";
  o->fixed_text = "the fix carves each uv context at 16 bytes per column";
  free(p);
}
