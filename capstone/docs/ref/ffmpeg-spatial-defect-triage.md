# FFmpeg: triage for SPATIAL defects, and the three places per-allocation bounds are blind

Mirror of `wireshark-spatial-defect-triage.md` for FFmpeg. The temporal hunt's filter 1 was
*"lifetime wording in the subject — 15 of 1,788 survive"* (`ffmpeg-live-defect-triage.md:26`), and
the same file records that the population was *measured* to be **spatial-dominated** — *"'overflow'
2,415, 'out of array' 1,062"* (`:51-54`) — counted in aggregate and never read case by case. So of
the three programs this is where the untriaged spatial population is largest by far.

## 1. The structural part, verified in upstream source first

FFmpeg has **three** distinct places where an overflow stays inside one allocation, so that bounds
at `av_malloc` granularity cannot see it. These are read from `n9.0.1`, not inferred, because the
whole classification depends on them.

### (a) Video frame planes are carved from ONE buffer — class C

`n9.0.1:libavutil/frame.c`, `get_video_buffer`:

```c
frame->buf[0] = av_buffer_alloc(total_size);                      /* :122 -- ONE allocation */
...
if ((ret = av_image_fill_pointers(frame->data, frame->format, padded_height,
                                  frame->buf[0]->data, frame->linesize)) < 0)   /* :128-129 */
```

Every plane pointer in `frame->data[]` is filled *into that single buffer*. **An overflow of plane Y
lands in plane U inside the same allocation.** A per-allocation bound is in-bounds for the whole
write; only a per-plane bound catches it. This is the richest class-C surface in any of the three
programs, and it is where FFmpeg's plane-indexing overflows live.

### (b) Audio planes are separate allocations — but each carries alignment slack

`n9.0.1:libavutil/frame.c`, `get_audio_buffer`:

```c
size = frame->linesize[0] + (size_t)align;                        /* :182 */
for (int i = 0; i < FFMIN(planes, AV_NUM_DATA_POINTERS); i++) {
    frame->buf[i] = av_buffer_alloc(size);                        /* :185 -- one PER channel */
    frame->extended_data[i] = frame->data[i] =
        (uint8_t *)FFALIGN((uintptr_t)frame->buf[i]->data, align);  /* :190-191 */
```

So an audio **channel-to-channel** overflow is class **A** — it crosses an `av_malloc` bound and
`shrink`/`sublet` already fault. But the allocation is `linesize[0] + align` while the usable plane
is `linesize[0]`, and `data[i]` is aligned *forward* into it. The bytes between the plane's logical
end and the allocation's end are **in bounds of the allocation and out of bounds of the plane**, so
a small overrun is class C even here. Worth stating because it means "audio is class A" is only true
for overflows large enough to clear the slack.

### (c) A refstruct object's header and payload are ONE allocation — class C

`n9.0.1:libavutil/refstruct.c`:

```c
buf = av_malloc(size + REFCOUNT_OFFSET);                          /* :109 -- ONE allocation */
...
RefCount *ref = (RefCount*)((char*)obj - REFCOUNT_OFFSET);        /* :72 */
```

Callers hold `obj`, a pointer *into* the allocation past the header. **An underflow from the payload
overwrites the RefCount**, inside the same `av_malloc`. This is the real mechanism behind the
project's synthetic fixture `ffapp` fx16 `rs_underflow` (*"one byte BELOW a refstruct object, where
stock keeps its RefCount"*), now confirmed upstream rather than assumed.

## 2. The port cannot measure any of this yet — and again the gap is wiring

Every FFmpeg port patch targets the **pool** APIs and nothing else:

- `ports/ffmpeg/sublet/ffmpeg-9.0.1-0001-avbufferpool-on-sublet.patch`
- `ports/ffmpeg/sublet/ffmpeg-9.0.1-0002-refstruct-on-sublet.patch`
- `ports/ffmpeg/buffer-pool/patches/*-pool-*.patch`

**No patch in the tree touches `av_buffer_alloc`, `frame.c` or `av_image_fill_pointers`** (checked
across every `.patch` under `ports/ffmpeg/`). So the port narrows pool payloads, not frame planes
and not the refstruct header boundary. A class-C plane defect would be caught by **no arm we have**.

That is the same shape as the `BLOCK_FAST` gap on the tshark side
(`wireshark-spatial-defect-triage.md`): the adapters narrow the allocator each port was built for,
and the real spatial defects sit one layer over. In both cases it is unwired coverage rather than a
design limit.

Note `refstruct` is a partial exception: the sublet patch does put refstruct *pool* objects under
leases, which is why fx16 discriminates on `pool0`/`pool2`/`poolsublet`. What it does not cover is a
refstruct object allocated outside a pool.

## 3. Candidates

**UNRESOLVED — the automated pass produced nothing usable, and that is an instrument result.**

`spatial-triage.py --program ffmpeg` over `n9.0.1..origin/master` (1,786 commits) kept **19** on
filter 1 and then classified **18 of 19 as `?`**, one as `STACK`, and **zero** as A, B or C. That is
not a finding about FFmpeg. The cause is specific and known: filter 2 locates an allocation site by
looking for the overflowed buffer's name assigned on a line with an allocator call, and in FFmpeg the
buffer is almost always `frame->data[i]` or a pool payload — allocated in `libavutil`, many call
frames away from the dissector-like code the fix touches. The one-alias-hop heuristic that worked on
Wireshark's `wmem_alloc(pinfo->pool, n)` cannot reach it.

So the FFmpeg candidate list needs a **shape** search, not a wording-plus-one-hop search — exactly
the move `ffmpeg-pool-consumer-defects.md:51-56` already had to make on the temporal side (*"the
usable instrument is a shape search over the consumer surface, cross-checked per candidate at the
source"*). The shapes to search for, from §1:

1. a fix that changes a bound on an index into `frame->data[i]` / `frame->linesize[i]`;
2. a fix that changes a `REFCOUNT_OFFSET`-relative access or a refstruct size;
3. a fix carrying a `Fixes: …clusterfuzz…` trailer whose diff touches plane indexing — about
   **2,993** commits carry such a trailer, which is the largest unexploited seam here.

**Until that search runs, FFmpeg contributes 0 triaged candidates.** It must not be recorded as
"FFmpeg has no class-B/C spatial defects"; §1 shows the mechanisms exist by construction.

## What this instrument cannot see

- Everything in §3: the allocation site when it is more than one alias hop from the fix.
- A defect never fixed upstream, and a fix that does not word itself spatially.
- Liveness is not asserted for any FFmpeg candidate here, because there are none. When there are,
  note that **filter 3 has a measured ~20% false-live rate** (`memcached-spatial-defect-triage.md`
  §5), so liveness must be confirmed by reading `n9.0.1:<path>` directly.
