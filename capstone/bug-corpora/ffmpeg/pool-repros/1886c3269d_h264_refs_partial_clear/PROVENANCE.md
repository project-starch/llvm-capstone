# Provenance

**Tier: LITERAL-traceable allocator, reduced consumer.** The pool is the real
`libavutil/buffer.c`. The reset and its bound are transcribed; the H.264 slice
decoder, its reference-list construction and the pictures themselves are reduced
to the array of records and the buffers they name.

- **Fix:** `1886c3269d` — *"avcodec/h264_refs: Clear stale pointers from ref_list"*, 2026-05-03. The `memset` length changes from `ref_count - len` to `32 - len`, in two places.
- **File:** `libavcodec/h264_refs.c`.
- **Reported by:** a browser vendor, with a proof-of-concept bitstream named in the upstream trailer.
- **CVE:** `NO VERIFIED CVE`. None in the commit; no advisory was searched for this entry.
- **Backport:** `8462c37595` on `release/8.1`.
- **Live at the pin:** no. Present at n8.0 and n8.1, gone from n8.1.1 onwards, so absent from 9.0.1. As with the other cases, the pin governs the fidelity tier, not whether the fixture builds.

## Why the stale storage is pool memory

`H264Ref.data[3]` points into the planes of an `H264Picture`, and those planes
come from `av_buffer_pool_get` through `avcodec_default_get_buffer2`
(`libavcodec/get_buffer.c:196`). A returned picture puts that storage on the
pool's freelist at the same address, which is what `reuse_same_address=1`
records.
