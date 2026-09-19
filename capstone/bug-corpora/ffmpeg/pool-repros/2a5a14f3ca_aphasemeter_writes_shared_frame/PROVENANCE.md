# Provenance

**Tier: LITERAL-traceable allocator and predicate, reduced consumer.** The pool,
`av_buffer_is_writable` and `av_buffer_get_ref_count` are the real ones from
`libavutil/buffer.c`. Reduced: the filter graph and the drawing.

- **Fix:** `2a5a14f3ca` — *"avfilter/avf_aphasemeter: make frame writable before writing to it"*, 2022-03-04. One line: `av_frame_make_writable(s->out);`.
- **File:** `libavfilter/avf_aphasemeter.c`.
- **CVE:** `NO VERIFIED CVE`.
- **Live at the pin:** no; fixed long before 9.0.1. The pin governs the fidelity tier, not whether the fixture builds.
- **Sibling reports:** `8061098418`, `2a5a14f3ca`, `de07c57d5a`, `faac31cc86`, `dc8e83b4e0` are five commits of one day adding the same missing check to five filters.

## Which pool the storage comes from

The retained frame comes through `ff_get_video_buffer` from
`libavfilter/framepool.c`, whose `FFFramePool` is a client of
`av_buffer_pool_init`/`av_buffer_pool_get` (`framepool.c:70,114,151`).
