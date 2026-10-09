# b5ff61695f — V is placed at the pre-doubling half stride, so 16-bit scaling's int32 U samples run into V

## The defect

V is placed at the pre-doubling half stride, so 16-bit scaling's int32 U samples run into V.

## Upstream defect

- **Fix:** `b5ff61695f` (`b5ff61695f2c0493036b463666c5936ee10da344`), subject: "sws: fix uv overwrite in 32bt".
- **Carved object:** each chrUPixBuf line, one FF_ALLOC_OR_GOTO(dst_stride * 2 + 1) at libswscale/utils.c:1053, carved into U and V (:1055).
- **Consumer:** libswscale/utils.c:789, :889-890 and :1053-1055 at the fix's parent, sws_init_context(); the writer is the horizontal chroma scaler.
- **The crossing:** writes 48 bytes past U's 208, into V; it stays inside the allocation.
- **Consequence:** V then overwrites U's tail, and the vertical scaler reads V's samples as U's.
- **Live at the `n9.0.1` pin:** NO — the carve moved and carries the fix's form at n9.0.1: libswscale/slice.c:63-68 allocates `size * 2 + 32` and places V at `line + size + 16`, where size is the stride AFTER the 16-bit doubling (slice.c:257-273, passed at :311). The pre-fix `chrUPixBuf[i] + dst_stride_px` occurs 0 times at the pin's libswscale/utils.c and once at the fix's parent.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libswscale/utils.c b/libswscale/utils.c
index 15d594c582..3e1662716b 100644
--- a/libswscale/utils.c
+++ b/libswscale/utils.c
@@ -786,7 +786,7 @@ int sws_init_context(SwsContext *c, SwsFilter *srcFilter, SwsFilter *dstFilter)
     int srcH= c->srcH;
     int dstW= c->dstW;
     int dstH= c->dstH;
-    int dst_stride = FFALIGN(dstW * sizeof(int16_t)+66, 16), dst_stride_px = dst_stride >> 1;
+    int dst_stride = FFALIGN(dstW * sizeof(int16_t)+66, 16);
     int flags, cpu_flags;
     enum PixelFormat srcFormat= c->srcFormat;
     enum PixelFormat dstFormat= c->dstFormat;
@@ -1047,12 +1047,12 @@ int sws_init_context(SwsContext *c, SwsFilter *srcFilter, SwsFilter *dstFilter)
         FF_ALLOCZ_OR_GOTO(c, c->lumPixBuf[i+c->vLumBufSize], dst_stride+1, fail);
         c->lumPixBuf[i] = c->lumPixBuf[i+c->vLumBufSize];
     }
-    c->uv_off = dst_stride_px;
+    c->uv_off = dst_stride>>1;
     c->uv_offx2 = dst_stride;
     for (i=0; i<c->vChrBufSize; i++) {
         FF_ALLOC_OR_GOTO(c, c->chrUPixBuf[i+c->vChrBufSize], dst_stride*2+1, fail);
         c->chrUPixBuf[i] = c->chrUPixBuf[i+c->vChrBufSize];
-        c->chrVPixBuf[i] = c->chrVPixBuf[i+c->vChrBufSize] = c->chrUPixBuf[i] + dst_stride_px;
+        c->chrVPixBuf[i] = c->chrVPixBuf[i+c->vChrBufSize] = c->chrUPixBuf[i] + (dst_stride>>1);
     }
     if (CONFIG_SWSCALE_ALPHA && c->alpPixBuf)
         for (i=0; i<c->vLumBufSize; i++) {
```
