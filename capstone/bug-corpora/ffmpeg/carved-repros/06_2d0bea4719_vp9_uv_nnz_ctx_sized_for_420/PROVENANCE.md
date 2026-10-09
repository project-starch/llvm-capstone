# 2d0bea4719 — a 4:4:4 frame clears 16 bytes per superblock column of a uv context carved for 4:2:0's 8

## The defect

A 4:4:4 frame clears 16 bytes per superblock column of a uv context carved for 4:2:0's 8.

## Upstream defect

- **Fix:** `2d0bea4719` (`2d0bea4719588aa9caa3f452596b9748ba13059e`), subject: "vp9: increase buffer sizes for non-420 chroma subsamplings.".
- **Carved object:** the update_size() block, one av_malloc(sb_cols * (240 + sizeof(lflvl) + 16 * sizeof(mv))) at libavcodec/vp9.c:329, carved by assign() at :331-347.
- **Consumer:** libavcodec/vp9.c:3877-3878 at the fix's parent, vp9_decode_frame().
- **The crossing:** clears 16 bytes per superblock column past above_uv_nnz_ctx[0], into above_uv_nnz_ctx[1]; it stays inside the allocation.
- **Consequence:** none: zeros, over a region the next statement clears anyway.
- **Live at the `n9.0.1` pin:** NO — n9.0.1:libavcodec/vp9.c contains `assign(s->above_uv_nnz_ctx[0], uint8_t *,             16);` once and the pre-fix 8-byte assign 0 times; 1/0 at the fix, 0/1 at its parent.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/vp9.c b/libavcodec/vp9.c
index bd852d4245..0b11bdc177 100644
--- a/libavcodec/vp9.c
+++ b/libavcodec/vp9.c
@@ -215,7 +215,7 @@ typedef struct VP9Context {
     DECLARE_ALIGNED(16, uint8_t, left_y_nnz_ctx)[16];
     DECLARE_ALIGNED(16, uint8_t, left_mode_ctx)[16];
     DECLARE_ALIGNED(16, VP56mv, left_mv_ctx)[16][2];
-    DECLARE_ALIGNED(8, uint8_t, left_uv_nnz_ctx)[2][8];
+    DECLARE_ALIGNED(16, uint8_t, left_uv_nnz_ctx)[2][16];
     DECLARE_ALIGNED(8, uint8_t, left_partition_ctx)[8];
     DECLARE_ALIGNED(8, uint8_t, left_skip_ctx)[8];
     DECLARE_ALIGNED(8, uint8_t, left_txfm_ctx)[8];
@@ -248,8 +248,8 @@ typedef struct VP9Context {
     int16_t *block_base, *block, *uvblock_base[2], *uvblock[2];
     uint8_t *eob_base, *uveob_base[2], *eob, *uveob[2];
     struct { int x, y; } min_mv, max_mv;
-    DECLARE_ALIGNED(32, uint8_t, tmp_y)[64*64];
-    DECLARE_ALIGNED(32, uint8_t, tmp_uv)[2][32*32];
+    DECLARE_ALIGNED(32, uint8_t, tmp_y)[64 * 64];
+    DECLARE_ALIGNED(32, uint8_t, tmp_uv)[2][64 * 64];
     uint16_t mvscale[3][2];
     uint8_t mvstep[3][2];
 } VP9Context;
@@ -307,39 +307,42 @@ static int vp9_ref_frame(AVCodecContext *ctx, VP9Frame *dst, VP9Frame *src)
     return 0;
 }
 
-static int update_size(AVCodecContext *ctx, int w, int h)
+static int update_size(AVCodecContext *ctx, int w, int h, enum AVPixelFormat fmt)
 {
     VP9Context *s = ctx->priv_data;
     uint8_t *p;
 
     av_assert0(w > 0 && h > 0);
 
-    if (s->intra_pred_data[0] && w == ctx->width && h == ctx->height)
+    if (s->intra_pred_data[0] && w == ctx->width && h == ctx->height && ctx->pix_fmt == fmt)
         return 0;
 
-    ctx->width  = w;
-    ctx->height = h;
-    s->sb_cols  = (w + 63) >> 6;
-    s->sb_rows  = (h + 63) >> 6;
-    s->cols     = (w + 7) >> 3;
-    s->rows     = (h + 7) >> 3;
+    ctx->width   = w;
+    ctx->height  = h;
+    ctx->pix_fmt = fmt;
+    s->sb_cols   = (w + 63) >> 6;
+    s->sb_rows   = (h + 63) >> 6;
+    s->cols      = (w + 7) >> 3;
+    s->rows      = (h + 7) >> 3;
 
 #define assign(var, type, n) var = (type) p; p += s->sb_cols * (n) * sizeof(*var)
     av_freep(&s->intra_pred_data[0]);
-    p = av_malloc(s->sb_cols * (240 + sizeof(*s->lflvl) + 16 * sizeof(*s->above_mv_ctx)));
+    // FIXME we slightly over-allocate here for subsampled chroma, but a little
+    // bit of padding shouldn't affect performance...
+    p = av_malloc(s->sb_cols * (320 + sizeof(*s->lflvl) + 16 * sizeof(*s->above_mv_ctx)));
     if (!p)
         return AVERROR(ENOMEM);
     assign(s->intra_pred_data[0],  uint8_t *,             64);
-    assign(s->intra_pred_data[1],  uint8_t *,             32);
-    assign(s->intra_pred_data[2],  uint8_t *,             32);
+    assign(s->intra_pred_data[1],  uint8_t *,             64);
+    assign(s->intra_pred_data[2],  uint8_t *,             64);
     assign(s->above_y_nnz_ctx,     uint8_t *,             16);
     assign(s->above_mode_ctx,      uint8_t *,             16);
     assign(s->above_mv_ctx,        VP56mv(*)[2],          16);
+    assign(s->above_uv_nnz_ctx[0], uint8_t *,             16);
+    assign(s->above_uv_nnz_ctx[1], uint8_t *,             16);
     assign(s->above_partition_ctx, uint8_t *,              8);
     assign(s->above_skip_ctx,      uint8_t *,              8);
     assign(s->above_txfm_ctx,      uint8_t *,              8);
-    assign(s->above_uv_nnz_ctx[0], uint8_t *,              8);
-    assign(s->above_uv_nnz_ctx[1], uint8_t *,              8);
     assign(s->above_segpred_ctx,   uint8_t *,              8);
     assign(s->above_intra_ctx,     uint8_t *,              8);
     assign(s->above_comp_ctx,      uint8_t *,              8);
@@ -358,34 +361,39 @@ static int update_size(AVCodecContext *ctx, int w, int h)
 static int update_block_buffers(AVCodecContext *ctx)
 {
     VP9Context *s = ctx->priv_data;
+    int chroma_blocks, chroma_eobs;
 
     if (s->b_base && s->block_base && s->block_alloc_using_2pass == s->frames[CUR_FRAME].uses_2pass)
         return 0;
 
     av_free(s->b_base);
     av_free(s->block_base);
+    chroma_blocks = 64 * 64 >> (s->ss_h + s->ss_v);
+    chroma_eobs   = 16 * 16 >> (s->ss_h + s->ss_v);
     if (s->frames[CUR_FRAME].uses_2pass) {
         int sbs = s->sb_cols * s->sb_rows;
 
         s->b_base = av_malloc_array(s->cols * s->rows, sizeof(VP9Block));
-        s->block_base = av_mallocz((64 * 64 + 128) * sbs * 3);
+        s->block_base = av_mallocz(((64 * 64 + 2 * chroma_blocks) * sizeof(int16_t) +
+                                    16 * 16 + 2 * chroma_eobs) * sbs);
         if (!s->b_base || !s->block_base)
             return AVERROR(ENOMEM);
         s->uvblock_base[0] = s->block_base + sbs * 64 * 64;
-        s->uvblock_base[1] = s->uvblock_base[0] + sbs * 32 * 32;
-        s->eob_base = (uint8_t *) (s->uvblock_base[1] + sbs * 32 * 32);
-        s->uveob_base[0] = s->eob_base + 256 * sbs;
-        s->uveob_base[1] = s->uveob_base[0] + 64 * sbs;
+        s->uvblock_base[1] = s->uvblock_base[0] + sbs * chroma_blocks;
+        s->eob_base = (uint8_t *) (s->uvblock_base[1] + sbs * chroma_blocks);
+        s->uveob_base[0] = s->eob_base + 16 * 16 * sbs;
+        s->uveob_base[1] = s->uveob_base[0] + chroma_eobs * sbs;
     } else {
         s->b_base = av_malloc(sizeof(VP9Block));
-        s->block_base = av_mallocz((64 * 64 + 128) * 3);
+        s->block_base = av_mallocz((64 * 64 + 2 * chroma_blocks) * sizeof(int16_t) +
+                                   16 * 16 + 2 * chroma_eobs);
         if (!s->b_base || !s->block_base)
             return AVERROR(ENOMEM);
         s->uvblock_base[0] = s->block_base + 64 * 64;
-        s->uvblock_base[1] = s->uvblock_base[0] + 32 * 32;
-        s->eob_base = (uint8_t *) (s->uvblock_base[1] + 32 * 32);
-        s->uveob_base[0] = s->eob_base + 256;
-        s->uveob_base[1] = s->uveob_base[0] + 64;
+        s->uvblock_base[1] = s->uvblock_base[0] + chroma_blocks;
+        s->eob_base = (uint8_t *) (s->uvblock_base[1] + chroma_blocks);
+        s->uveob_base[0] = s->eob_base + 16 * 16;
+        s->uveob_base[1] = s->uveob_base[0] + chroma_eobs;
     }
     s->block_alloc_using_2pass = s->frames[CUR_FRAME].uses_2pass;
 
@@ -772,8 +780,8 @@ static int decode_frame_header(AVCodecContext *ctx,
     }
 
     /* tiling info */
-    if ((res = update_size(ctx, w, h)) < 0) {
-        av_log(ctx, AV_LOG_ERROR, "Failed to initialize decoder for %dx%d\n", w, h);
+    if ((res = update_size(ctx, w, h, fmt)) < 0) {
+        av_log(ctx, AV_LOG_ERROR, "Failed to initialize decoder for %dx%d @ %d\n", w, h, fmt);
         return res;
     }
     for (s->tiling.log2_tile_cols = 0;
@@ -3960,7 +3968,7 @@ static int vp9_decode_frame(AVCodecContext *ctx, void *frame,
                             memset(s->left_mode_ctx, NEARESTMV, 8);
                         }
                         memset(s->left_y_nnz_ctx, 0, 16);
-                        memset(s->left_uv_nnz_ctx, 0, 16);
+                        memset(s->left_uv_nnz_ctx, 0, 32);
                         memset(s->left_segpred_ctx, 0, 8);
 
                         memcpy(&s->c, &s->c_b[tile_col], sizeof(s->c));
@@ -4089,7 +4097,6 @@ static av_cold int vp9_decode_init(AVCodecContext *ctx)
     VP9Context *s = ctx->priv_data;
 
     ctx->internal->allocate_progress = 1;
-    ctx->pix_fmt = AV_PIX_FMT_YUV420P;
     ff_vp9dsp_init(&s->dsp);
     ff_videodsp_init(&s->vdsp, 8);
     s->filter.sharpness = -1;
```
