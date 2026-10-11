# Provenance

**Upstream fix:** `316531e61c`, *"avfilter/vidstabtransform: always use in-place
transform path"*, 2026-04-01. **Not live at the 9.0.1 pin** — the removed lines
are present from `n8.0` through `n8.1.2` and gone from 9.0.x.

**Consumer:** `libavfilter/vf_vidstabtransform.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

The upstream message states the mechanism, and the fixture is that paragraph:

> The separate-buffer path stores a shallow copy of the source frame pointer in
> `td->src` without allocating internal memory (`srcMalloced` stays 0). When a
> subsequent frame takes the in-place path, `vsFrameIsNull(&td->src)` is false
> so `vsFrameAllocate()` is skipped, and `vsFrameCopy()` writes into the stale
> pointer left over from the previous frame, corrupting memory that the caller
> no longer owns.

The fix removes the choice between paths so the library never keeps a shallow
copy. This is the corpus's only **write** probe: the stale reference does not
read the new owner's data, it overwrites it. The CPython corpus files this
shape as *parked with no bound on reuse*; what is distinctive here is where it
is parked — inside an opaque third-party library's state, where no amount of
care in FFmpeg's own code would find it.

## Why the storage is pool storage

The source frame is obtained through `ff_get_video_buffer` from
`libavfilter/framepool.c`, whose `FFFramePool` is a client of
`av_buffer_pool_init`/`av_buffer_pool_get` (`framepool.c:70,114,151`).

## Real and reduced

Real: `libavutil/buffer.c`. Reduced, and this is the largest reduction in this
corpus and is stated rather than implied: `libvidstab` is not linked, and its
state is modelled as the one field that matters, the retained `src` pointer.
What that preserves is the property under test — a raw pointer into pooled
storage, held across a frame boundary by code that never took a reference. What
it drops is every other reason `vsTransformPrepare` might touch that storage.
The real filter needs `--enable-libvidstab`; the fixture does not, because the
retained pointer is the defect.
