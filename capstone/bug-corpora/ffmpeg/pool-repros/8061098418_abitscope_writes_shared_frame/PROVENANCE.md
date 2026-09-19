# Provenance

**Tier: LITERAL-traceable allocator and predicate, reduced consumer.** The pool,
`av_buffer_is_writable` and `av_buffer_get_ref_count` are the real ones from
`libavutil/buffer.c`. Reduced: the filter graph, the audio analysis and the
drawing itself, none of which changes which storage is written or who else
holds it.

- **Fix:** `8061098418` — *"avfilter/avf_abitscope: make frame writable before writing to it"*, 2022-03-04. One line: `av_frame_make_writable(s->outpicref);` before the clone.
- **File:** `libavfilter/avf_abitscope.c`, `filter_frame()`.
- **CVE:** `NO VERIFIED CVE`.
- **Live at the pin:** no — fixed long before 9.0.1. As with the other cases the pin governs the fidelity tier, not whether the fixture builds.

## Which pool the storage comes from

`s->outpicref` is obtained through `ff_get_video_buffer`, which draws from
`libavfilter/framepool.c`; that file's `FFFramePool` is a client of
`av_buffer_pool_init`/`av_buffer_pool_get` (`framepool.c:70,114,151`), so the
storage is buffer-pool storage and its reuse is the pool's LIFO reissue.
