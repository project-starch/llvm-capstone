# d2213b6493 — after a resolution change the B-frame scratch stays carved at the old linesize, so direction 0's prediction runs into direction 1's

## The defect

After a resolution change the B-frame scratch stays carved at the old linesize, so direction 0's prediction runs into direction 1's.

## Upstream defect

- **Fix:** `d2213b6493` (`d2213b649383bc4d9428af42943457f782493be0`), subject: "rv34: Fix buffer size used for MC of B frames after a resolution change".
- **Carved object:** r->tmp_b_block_base, one av_malloc(linesize * 48) at libavcodec/rv34.c:1315, carved into tmp_b_block_y[0..1] and tmp_b_block_uv[0..3] (:1316-1320).
- **Consumer:** libavcodec/rv34.c:789 at the fix's parent, rv34_mc(), weighted bidirectional; the dead re-carve guard at :1311.
- **The crossing:** writes direction 0's rows 8..15 onto tmp_b_block_y[1] after the linesize doubled; it stays inside the allocation.
- **Consequence:** direction 1 overwrites them, and rv4_weight averages direction 1 with itself.
- **Live at the `n9.0.1` pin:** NO — n9.0.1:libavcodec/rv34.c contains `if (!r->tmp_b_block_base) {` once and the dead guard `if (!r->tmp_b_block_base || s->width != r->si.width || s->height != r->si.height) {` 0 times; 1/0 at the fix, 0/1 at its parent.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/rv34.c b/libavcodec/rv34.c
index aa52a93dbb..5ec8bb369c 100644
--- a/libavcodec/rv34.c
+++ b/libavcodec/rv34.c
@@ -1303,15 +1303,15 @@ static int rv34_decode_slice(RV34DecContext *r, int end, const uint8_t* buf, int
             r->cbp_luma   = av_realloc(r->cbp_luma,   r->s.mb_stride * r->s.mb_height * sizeof(*r->cbp_luma));
             r->cbp_chroma = av_realloc(r->cbp_chroma, r->s.mb_stride * r->s.mb_height * sizeof(*r->cbp_chroma));
             r->deblock_coefs = av_realloc(r->deblock_coefs, r->s.mb_stride * r->s.mb_height * sizeof(*r->deblock_coefs));
+            av_freep(&r->tmp_b_block_base);
         }
         s->pict_type = r->si.type ? r->si.type : AV_PICTURE_TYPE_I;
         if(MPV_frame_start(s, s->avctx) < 0)
             return -1;
         ff_er_frame_start(s);
-        if (!r->tmp_b_block_base || s->width != r->si.width || s->height != r->si.height) {
+        if (!r->tmp_b_block_base) {
             int i;
 
-            av_free(r->tmp_b_block_base); //realloc() doesn't guarantee alignment
             r->tmp_b_block_base = av_malloc(s->linesize * 48);
             for (i = 0; i < 2; i++)
                 r->tmp_b_block_y[i] = r->tmp_b_block_base + i * 16 * s->linesize;
```
