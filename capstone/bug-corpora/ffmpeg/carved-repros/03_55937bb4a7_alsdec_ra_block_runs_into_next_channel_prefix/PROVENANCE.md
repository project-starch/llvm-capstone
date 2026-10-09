# 55937bb4a7 — the RA block reconstructs opt_order samples of a shorter block, decrementing the next channel's carry-over prefix

## The defect

The RA block reconstructs opt_order samples of a shorter block, decrementing the next channel's carry-over prefix.

## Upstream defect

- **Fix:** `55937bb4a7` (`55937bb4a7df157fb08f79e7e623a16280533275`), subject: "libavcodec/als: fix address sanitization error in decoder".
- **Carved object:** ctx->raw_buffer, one av_mallocz_array(channels * (frame_length + max_order)) at libavcodec/alsdec.c:2059, carved per channel at :2096-2098.
- **Consumer:** libavcodec/alsdec.c:922-931 at the fix's parent, decode_var_block_data(), the random-access block.
- **The crossing:** read-modify-writes 8 int32 past channel 0's samples, into channel 1's carry-over prefix; it stays inside the allocation.
- **Consequence:** channel 1's next prediction starts from a decremented sample.
- **Live at the `n9.0.1` pin:** NO — n9.0.1:libavcodec/alsdec.c contains `for (smp = 0; smp < FFMIN(opt_order, block_length); smp++) {` once and the pre-fix `for (smp = 0; smp < opt_order; smp++) {` 0 times; 1/0 at the fix, 0/1 at its parent.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/alsdec.c b/libavcodec/alsdec.c
index 9a72686413..ca8701e6d0 100644
--- a/libavcodec/alsdec.c
+++ b/libavcodec/alsdec.c
@@ -920,7 +920,7 @@ static int decode_var_block_data(ALSDecContext *ctx, ALSBlockData *bd)
 
     // reconstruct all samples from residuals
     if (bd->ra_block) {
-        for (smp = 0; smp < opt_order; smp++) {
+        for (smp = 0; smp < FFMIN(opt_order, block_length); smp++) {
             y = 1 << 19;
 
             for (sb = 0; sb < smp; sb++)
```
