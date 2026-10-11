# cd7524fdd1 — long_filter_high_3800 primes its delay line from `order` samples of a shorter channel, reading into decoded[1]

## The defect

Long_filter_high_3800 primes its delay line from `order` samples of a shorter channel, reading into decoded[1].

## Upstream defect

- **Fix:** `cd7524fdd1` (`cd7524fdd13dc8d0cf22e2cfd8300a245542b13a`), subject: "avcodec/apedec: Check length in long_filter_high_3800()".
- **Carved object:** s->decoded_buffer, carved into decoded[0] and decoded[1] exactly as case 1.
- **Consumer:** libavcodec/apedec.c:889-897 at the fix's parent, long_filter_high_3800(), called from predictor_decode_mono_3800() :1006.
- **The crossing:** reads 56 int32 past a 72-sample decoded[0], from decoded[1]; it stays inside the allocation.
- **Consequence:** none: the loop that would use the delay line does not run when order >= length.
- **Live at the `n9.0.1` pin:** NO — n9.0.1:libavcodec/apedec.c contains `if (order >= length)\n        return;` once and the pre-fix adjacency (the delay-line memset directly after the declarations) 0 times; 1/0 at the fix, 0/1 at its parent.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/apedec.c b/libavcodec/apedec.c
index fcccfbe6d4..a528e603a8 100644
--- a/libavcodec/apedec.c
+++ b/libavcodec/apedec.c
@@ -892,6 +892,9 @@ static void long_filter_high_3800(int32_t *buffer, int order, int shift, int len
     int32_t dotprod, sign;
     int32_t coeffs[256], delay[256];
 
+    if (order >= length)
+        return;
+
     memset(coeffs, 0, order * sizeof(*coeffs));
     for (i = 0; i < order; i++)
         delay[i] = buffer[i];
```
