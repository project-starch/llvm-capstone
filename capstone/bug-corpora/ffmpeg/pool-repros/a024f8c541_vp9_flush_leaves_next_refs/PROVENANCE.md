# Provenance

**Tier: allocator literal, consumer reduced, threading modelled.** The pool is
the real `libavutil/buffer.c`. The two thread contexts are modelled as two
arrays, because the hand-off is a copy between contexts and not a race; the
single-hart domain cannot run the original either way.

- **Fix:** `a024f8c541` on master — *"avcodec/vp9: unref next_refs in vp9_decode_flush"*, 2026-09-12; backported to `release/9.0` as `c878aa71a6`.
- **File:** `libavcodec/vp9.c`.
- **CVE:** `NO VERIFIED CVE`.
- **Live at the pin:** **yes.** The fix reached `release/9.0` after the n9.0.1 tag, so the defect is in the pinned tree. This is the only one of the four for which that holds.

## Which pool the storage comes from

`next_refs[]` holds `ProgressFrame` references whose internals come from
`progress_frame_pool`, an `av_refstruct_pool` created in
`libavcodec/decode.c:2136`, and the frames they name carry planes from the
buffer pool. The fixture models the reference discipline with the buffer pool
alone, since what the case turns on is which references a flush ends, not which
of the two pools backs them.
