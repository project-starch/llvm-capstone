# 699341d647 — decode_array_0000 writes 64 outputs whatever the frame length, so a short channel's write runs into decoded[1]

## The defect

Decode_array_0000 writes 64 outputs whatever the frame length, so a short channel's write runs into decoded[1].

## Upstream defect

- **Fix:** `699341d647` (`699341d647f7af785fb8ceed67604467b0b9ab12`), subject: "apedec: prevent out of array writes in decode_array_0000".
- **Carved object:** s->decoded_buffer, one av_fast_malloc of 2 * FFALIGN(blockstodecode, 8) int32 at libavcodec/apedec.c:1485-1486, carved into decoded[0] and decoded[1] (:1490-1491).
- **Consumer:** libavcodec/apedec.c:595-608 at the fix's parent, decode_array_0000(), called from entropy_decode_mono_0000() :632-636.
- **The crossing:** writes 24 int32 past a 40-sample decoded[0], into decoded[1]; it stays inside the allocation.
- **Consequence:** none on the mono path: decoded[1] is not read there.
- **Live at the `n9.0.1` pin:** NO — n9.0.1:libavcodec/apedec.c contains `for (; i < FFMIN(blockstodecode, 64); i++) {` once and the pre-fix `for (; i < 64; i++) {` 0 times; 1/0 at the fix, 0/1 at its parent.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/apedec.c b/libavcodec/apedec.c
index ffd54c10f8..03afd756dc 100644
--- a/libavcodec/apedec.c
+++ b/libavcodec/apedec.c
@@ -592,14 +592,14 @@ static void decode_array_0000(APEContext *ctx, GetBitContext *gb,
     int ksummax, ksummin;
 
     rice->ksum = 0;
-    for (i = 0; i < 5; i++) {
+    for (i = 0; i < FFMIN(blockstodecode, 5); i++) {
         out[i] = get_rice_ook(&ctx->gb, 10);
         rice->ksum += out[i];
     }
     rice->k = av_log2(rice->ksum / 10) + 1;
     if (rice->k >= 24)
         return;
-    for (; i < 64; i++) {
+    for (; i < FFMIN(blockstodecode, 64); i++) {
         out[i] = get_rice_ook(&ctx->gb, rice->k);
         rice->ksum += out[i];
         rice->k = av_log2(rice->ksum / ((i + 1) * 2)) + 1;
```
