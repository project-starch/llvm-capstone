# 043bcdcdb0 — the inter prediction at temp + 16 with a stride under 32 ends past temp's 16 rows, overwriting the source block

## The defect

The inter prediction at temp + 16 with a stride under 32 ends past temp's 16 rows, overwriting the source block.

## Upstream defect

- **Fix:** `043bcdcdb0` (`043bcdcdb00ebcbae80d7a9f78b763b33b9f0d15`), subject: "avcodec/svq1enc: fix encoding of small widths".
- **Carved object:** s->scratchbuf, one av_malloc(linesize[0] * 16 * 2) at libavcodec/svq1enc.c:589, carved into temp (:369) and src (+ stride * 16, :255).
- **Consumer:** libavcodec/svq1enc.c:430-435 at the fix's parent, svq1_encode_plane().
- **The crossing:** writes the prediction's last row, 16 bytes, onto src's first; it stays inside the allocation.
- **Consequence:** encode_block() then encodes against a corrupted source row.
- **Live at the `n9.0.1` pin:** NO — n9.0.1:libavcodec/svq1enc.c contains `s->scratchbuf + stride * 32;` once and the pre-fix `s->scratchbuf + stride * 16;` 0 times; 1/0 at the fix, 0/1 at its parent.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/svq1enc.c b/libavcodec/svq1enc.c
index 7631f053da..26e7aeaa9c 100644
--- a/libavcodec/svq1enc.c
+++ b/libavcodec/svq1enc.c
@@ -252,7 +252,7 @@ static int svq1_encode_plane(SVQ1EncContext *s, int plane,
     int block_width, block_height;
     int level;
     int threshold[6];
-    uint8_t *src     = s->scratchbuf + stride * 16;
+    uint8_t *src     = s->scratchbuf + stride * 32;
     const int lambda = (f->quality * f->quality) >>
                        (2 * FF_LAMBDA_SHIFT);
 
@@ -427,12 +427,12 @@ static int svq1_encode_plane(SVQ1EncContext *s, int plane,
 
                     dxy = (mx & 1) + 2 * (my & 1);
 
-                    s->hdsp.put_pixels_tab[0][dxy](temp + 16,
+                    s->hdsp.put_pixels_tab[0][dxy](temp + 16*stride,
                                                    ref + (mx >> 1) +
                                                    stride * (my >> 1),
                                                    stride, 16);
 
-                    score[1] += encode_block(s, src + 16 * x, temp + 16,
+                    score[1] += encode_block(s, src + 16 * x, temp + 16*stride,
                                              decoded, stride, 5, 64, lambda, 0);
                     best      = score[1] <= score[0];
 
@@ -586,7 +586,7 @@ static int svq1_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
             (ret = ff_get_buffer(avctx, s->last_picture, 0))   < 0) {
             return ret;
         }
-        s->scratchbuf = av_malloc(s->current_picture->linesize[0] * 16 * 2);
+        s->scratchbuf = av_malloc(s->current_picture->linesize[0] * 16 * 3);
     }
 
     FFSWAP(AVFrame*, s->current_picture, s->last_picture);
```
