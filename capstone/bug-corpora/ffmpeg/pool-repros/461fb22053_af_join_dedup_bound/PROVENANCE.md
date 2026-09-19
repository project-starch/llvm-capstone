# Provenance

**Tier: LITERAL-traceable allocator, reduced consumer.** The pool is the real
`libavutil/buffer.c`, compiled unmodified through the port. The dedup loop is
transcribed from upstream, including the defective comparison. Reduced: the
filter graph, two audio inputs and `ff_filter_frame`, none of which changes what
the allocator is asked to do, and the part of `AVFrame` the loop touches
(`extended_data`, `buf`), modelled because the port extracts the allocator and
not `libavutil/frame.c`.

## Upstream defect

- **Fix:** `461fb22053` — *"avfilter/af_join: fix wrong loop bound in buffer dedup (use-after-free)"*, 2026-05-25. One line: `if (j == i)` becomes `if (j == nb_buffers)`.
- **File:** `libavfilter/af_join.c`, `try_push_frame()`.
- **Reported by:** two external security researchers, named in the upstream trailer and deliberately not repeated here.
- **CVE:** `NO VERIFIED CVE`. The commit carries none, and no advisory was searched for this entry.
- **Backport:** `64fd93e361` on `release/8.1`.

## Live at which pin

Checked mechanically, by searching the fix's removed line verbatim in the tag's
own copy of the file:

| n8.0 | n8.1 | n8.1.1 | n8.1.2 | n9.0.1 |
|---|---|---|---|---|
| present | present | present | gone | gone |

**Not live at the port's current 9.0.1 pin** — and that does not block the
fixture. What is pinned is the *allocator*; the defective consumer loop is
transcribed here, so the case builds and runs against the 9.0.1 pool exactly as
it stands, which is what `results/` records. Liveness is a fidelity property, not
a build requirement: it decides whether the pinned tree still contains the
defect, and therefore whether this is a reduction of shipping code or of code
that shipped until n8.1.1. Moving the pin to n8.1 would raise that tier; nothing
else here depends on it.

## Why the stale storage is pool memory

`av_frame_get_plane_buffer()` returns the `AVBufferRef` backing that plane. For
frames the decoder produced, that reference came from `av_buffer_pool_get()` via
`avcodec_default_get_buffer2` (`libavcodec/get_buffer.c:196`), so the last unref
runs `pool_release_buffer` and the payload goes onto the pool's freelist at the
same address rather than to `free`. The fixture reproduces exactly that: the
pool reissues the identical pointer, which is what `reuse_same_address=1`
records.
