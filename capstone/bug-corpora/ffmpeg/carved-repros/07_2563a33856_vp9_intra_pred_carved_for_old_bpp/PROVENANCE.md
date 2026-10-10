# 2563a33856 — a frame thread decodes at 10-bit into intra-prediction regions carved for 8-bit, running intra_pred_data[0] into [1]

## The defect

A frame thread decodes at 10-bit into intra-prediction regions carved for 8-bit, running intra_pred_data[0] into [1].

## Upstream defect

- **Fix:** `2563a33856` (`2563a33856eb597c9d53b4c7cab07b6f18417740`), subject: "vp9: re-initialize internal buffers on bpp change also.".
- **Carved object:** the update_size() block, one av_malloc(sb_cols * (128 + 192 * bytesperpixel + ...)) at libavcodec/vp9.c:335-336, carved by assign() at :339-355 at the bit depth of the moment.
- **Consumer:** libavcodec/vp9.c:4207-4209 at the fix's parent, vp9_decode_frame(); the stale carve from vp9_decode_update_thread_context() :4321-4351.
- **The crossing:** writes 128 bytes past a 128-byte intra_pred_data[0], into [1]; it stays inside the allocation.
- **Consequence:** the next superblock row's luma intra prediction reads chroma samples.
- **Live at the `n9.0.1` pin:** NO — the mechanism is gone at n9.0.1, read rather than inferred from ancestry: vp9_decode_update_thread_context (n9.0.1:libavcodec/vp9.c:1875-1912) no longer frees or carves anything, and update_size re-carves whenever the context's pix_fmt differs from the format it last carved for (`s->pix_fmt == s->last_fmt`, :272-275), which a bit-depth change always does. The pre-fix guard `s->cols != ssrc->cols || s->rows != ssrc->rows)) {` occurs 0 times at the pin and once at the fix's parent; `s->pix_fmt == s->last_fmt` occurs once at the pin.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/vp9.c b/libavcodec/vp9.c
index fcdd6e128e..c605c08288 100644
--- a/libavcodec/vp9.c
+++ b/libavcodec/vp9.c
@@ -4319,7 +4319,8 @@ static int vp9_decode_update_thread_context(AVCodecContext *dst, const AVCodecCo
 
     // detect size changes in other threads
     if (s->intra_pred_data[0] &&
-        (!ssrc->intra_pred_data[0] || s->cols != ssrc->cols || s->rows != ssrc->rows)) {
+        (!ssrc->intra_pred_data[0] || s->cols != ssrc->cols ||
+         s->rows != ssrc->rows || s->bpp != ssrc->bpp)) {
         free_buffers(s);
     }
```
