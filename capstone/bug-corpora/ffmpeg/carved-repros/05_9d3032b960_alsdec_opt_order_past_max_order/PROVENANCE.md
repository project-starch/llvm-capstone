# 9d3032b960 — the adaptive order is read in a field sized from max_order + 1, so opt_order 31 writes 11 coefficients into the next channel's slice

## The defect

The adaptive order is read in a field sized from max_order + 1, so opt_order 31 writes 11 coefficients into the next channel's slice.

## Upstream defect

- **Fix:** `9d3032b960` (`9d3032b960ae03066c008d6e6774f68b17a1d69d`), subject: "alsdec: check opt_order.".
- **Carved object:** ctx->quant_cof_buffer, one av_malloc(num_buffers * max_order int32) at libavcodec/alsdec.c:1632-1633, carved per channel at :1648 (MCC mode).
- **Consumer:** libavcodec/alsdec.c:663-711 at the fix's parent, read_var_block_data().
- **The crossing:** writes 11 int32 past channel 0's 20-word slice, into channel 1's; it stays inside the allocation.
- **Consequence:** channel 0's prediction then uses channel 1's coefficients 20..30.
- **Live at the `n9.0.1` pin:** NO — n9.0.1:libavcodec/alsdec.c contains `if (*bd->opt_order > sconf->max_order) {` once and the pre-fix adjacency (`get_bits(gb, opt_order_length);` directly followed by `} else {`) 0 times; 1/0 at the fix, 0/1 at its parent.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/alsdec.c b/libavcodec/alsdec.c
index 63e58ff55a..a9b04b8569 100644
--- a/libavcodec/alsdec.c
+++ b/libavcodec/alsdec.c
@@ -663,6 +663,10 @@ static int read_var_block_data(ALSDecContext *ctx, ALSBlockData *bd)
             int opt_order_length = av_ceil_log2(av_clip((bd->block_length >> 3) - 1,
                                                 2, sconf->max_order + 1));
             *bd->opt_order       = get_bits(gb, opt_order_length);
+            if (*bd->opt_order > sconf->max_order) {
+                av_log(avctx, AV_LOG_ERROR, "Order too large\n");
+                return -1;
+            }
         } else {
             *bd->opt_order = sconf->max_order;
         }
```
