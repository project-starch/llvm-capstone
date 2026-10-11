# Provenance

**Upstream fix:** `1886c3269d`, *"avcodec/h264_refs: Clear stale pointers from
ref_list"*, 2026-05-03, backported to `release/8.1` as `8462c37595`. **Not live
at the 9.0.1 pin** — the removed lines are present in `n8.0` and `n8.1` and gone
from `n8.1.1` onwards.

**Consumer:** `libavcodec/h264_refs.c`. **CVE:** `NO VERIFIED CVE`. Reported
with a proof-of-concept bitstream by a browser vendor.

## The defect

`H264Ref` carries `data[3]`, pointers into a picture's planes, and a `parent`.
When the reference list shrinks, the reset clears entries `[len, ref_count)`.
The fix changes the length to cover the array instead:

```c
-            if (len < sl->ref_count[list])
-                memset(&sl->ref_list[list][len], 0, sizeof(H264Ref) * (sl->ref_count[list] - len));
+            memset(&sl->ref_list[list][len], 0, sizeof(H264Ref) * (32 - len));
```

Every record from `ref_count` to the array's 32 slots keeps whatever it held,
and those pictures have since been returned to the pool.

## Why the storage is pool storage

`H264Ref.data[3]` points into the planes of an `H264Picture`, and those planes
come from `av_buffer_pool_get` through `avcodec_default_get_buffer2`
(`libavcodec/get_buffer.c:196`). A returned picture puts that storage on the
pool's freelist at the same address, which the fixture records as
`reuse_same_address=1`.

## Real and reduced

Real: `libavutil/buffer.c`. Transcribed: the reset and its bound. Reduced: the
slice decoder, its reference-list construction and the pictures themselves, to
the array of records and the buffers they name.
