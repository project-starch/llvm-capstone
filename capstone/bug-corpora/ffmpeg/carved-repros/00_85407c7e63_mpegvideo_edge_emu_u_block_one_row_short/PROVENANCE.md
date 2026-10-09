# 85407c7e63 — interlaced MPEG-4 edge emulation writes ten U rows into a nine-row carve, so U's last row is V's first

## The defect

Interlaced MPEG-4 edge emulation writes ten U rows into a nine-row carve, so U's last row is V's first.

## Upstream defect

- **Fix:** `85407c7e63` (`85407c7e63722a2d723257e8cf5f281a8c9f34a4`), subject: "avcodec/mpegvideo: Fix edge emu buffer overlap with interlaced mpeg4".
- **Carved object:** sc->edge_emu_buffer, one FF_ALLOCZ_ARRAY_OR_GOTO(alloc_size, 4 * 68) at libavcodec/mpegpicture.c:79, carved into luma rows, ubuf (+18 * linesize) and vbuf (ubuf + 9 * uvlinesize).
- **Consumer:** libavcodec/mpegvideo_motion.c:328-340 at the fix's parent, mpeg_motion_internal().
- **The crossing:** writes U's tenth row, 9 bytes, onto vbuf's row 0; it stays inside the allocation.
- **Consequence:** V overwrites it; the field MC reads V's samples as U's.
- **Live at the `n9.0.1` pin:** NO — n9.0.1:libavcodec/mpegvideo_motion.c contains `vbuf = ubuf + 10 * s->uvlinesize` twice and the pre-fix `vbuf = ubuf + 9 * s->uvlinesize` 0 times; the same pair reads 2/0 at the fix and 0/2 at its parent, so the probe separates the two.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/mpegpicture.c b/libavcodec/mpegpicture.c
index 6748fc2986..302f9d20d3 100644
--- a/libavcodec/mpegpicture.c
+++ b/libavcodec/mpegpicture.c
@@ -76,7 +76,7 @@ int ff_mpeg_framesize_alloc(AVCodecContext *avctx, MotionEstContext *me,
     // at uvlinesize. It supports only YUV420 so 24x24 is enough
     // linesize * interlaced * MBsize
     // we also use this buffer for encoding in encode_mb_internal() needig an additional 32 lines
-    FF_ALLOCZ_ARRAY_OR_GOTO(avctx, sc->edge_emu_buffer, alloc_size, 4 * 68,
+    FF_ALLOCZ_ARRAY_OR_GOTO(avctx, sc->edge_emu_buffer, alloc_size, 4 * 70,
                       fail);
 
     FF_ALLOCZ_ARRAY_OR_GOTO(avctx, me->scratchpad, alloc_size, 4 * 16 * 2,
diff --git a/libavcodec/mpegvideo_motion.c b/libavcodec/mpegvideo_motion.c
index c29810f598..b97a6cb303 100644
--- a/libavcodec/mpegvideo_motion.c
+++ b/libavcodec/mpegvideo_motion.c
@@ -326,7 +326,7 @@ void mpeg_motion_internal(MpegEncContext *s,
         ptr_y = s->sc.edge_emu_buffer;
         if (!CONFIG_GRAY || !(s->avctx->flags & AV_CODEC_FLAG_GRAY)) {
             uint8_t *ubuf = s->sc.edge_emu_buffer + 18 * s->linesize;
-            uint8_t *vbuf = ubuf + 9 * s->uvlinesize;
+            uint8_t *vbuf = ubuf + 10 * s->uvlinesize;
             uvsrc_y = (unsigned)uvsrc_y << field_based;
             s->vdsp.emulated_edge_mc(ubuf, ptr_cb,
                                      s->uvlinesize, s->uvlinesize,
@@ -549,7 +549,7 @@ static inline void qpel_motion(MpegEncContext *s,
         ptr_y = s->sc.edge_emu_buffer;
         if (!CONFIG_GRAY || !(s->avctx->flags & AV_CODEC_FLAG_GRAY)) {
             uint8_t *ubuf = s->sc.edge_emu_buffer + 18 * s->linesize;
-            uint8_t *vbuf = ubuf + 9 * s->uvlinesize;
+            uint8_t *vbuf = ubuf + 10 * s->uvlinesize;
             s->vdsp.emulated_edge_mc(ubuf, ptr_cb,
                                      s->uvlinesize, s->uvlinesize,
                                      9, 9 + field_based,
```
