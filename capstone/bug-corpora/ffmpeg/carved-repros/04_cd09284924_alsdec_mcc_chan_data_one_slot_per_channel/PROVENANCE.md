# cd09284924 — multi-channel correlation carves one slot per channel but reads a list per channel, so channel 0's terminator lands in channel 1's slot

## The defect

Multi-channel correlation carves one slot per channel but reads a list per channel, so channel 0's terminator lands in channel 1's slot.

## Upstream defect

- **Fix:** `cd09284924` (`cd0928492410c5a93959d664362cd0d0ee50b961`), subject: "Fix wrong buffer allocation for MCC in ALS.".
- **Carved object:** ctx->chan_data_buffer, one av_malloc(sizeof(ALSChannelData) * num_buffers) at libavcodec/alsdec.c:1565-1566, carved one slot per channel at :1579.
- **Consumer:** libavcodec/alsdec.c:1119-1126 at the fix's parent, read_channel_data().
- **The crossing:** writes channel 0's terminating stop_flag into channel 1's 44-byte slot; it stays inside the allocation.
- **Consequence:** channel 1's entry replaces the terminator, so channel 0's dependency list runs into channel 1's.
- **Live at the `n9.0.1` pin:** NO — n9.0.1:libavcodec/alsdec.c contains `ctx->chan_data[c] = ctx->chan_data_buffer + c * num_buffers;` once and the pre-fix `ctx->chan_data[c] = ctx->chan_data_buffer + c;` 0 times; 1/0 at the fix, 0/1 at its parent.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/alsdec.c b/libavcodec/alsdec.c
index b9a22850cb..25b61ec0ab 100644
--- a/libavcodec/alsdec.c
+++ b/libavcodec/alsdec.c
@@ -1563,7 +1563,7 @@ static av_cold int decode_init(AVCodecContext *avctx)
     // allocate and assign channel data buffer for mcc mode
     if (sconf->mc_coding) {
         ctx->chan_data_buffer  = av_malloc(sizeof(*ctx->chan_data_buffer) *
-                                           num_buffers);
+                                           num_buffers * num_buffers);
         ctx->chan_data         = av_malloc(sizeof(ALSChannelData) *
                                            num_buffers);
         ctx->reverted_channels = av_malloc(sizeof(*ctx->reverted_channels) *
@@ -1576,7 +1576,7 @@ static av_cold int decode_init(AVCodecContext *avctx)
         }
 
         for (c = 0; c < num_buffers; c++)
-            ctx->chan_data[c] = ctx->chan_data_buffer + c;
+            ctx->chan_data[c] = ctx->chan_data_buffer + c * num_buffers;
     } else {
         ctx->chan_data         = NULL;
         ctx->chan_data_buffer  = NULL;
```
