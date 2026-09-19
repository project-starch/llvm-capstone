# FFmpeg pool-consumer defects — triage at the 9.0.1 pin

*Which upstream FFmpeg defects are usable as specimens for the nested-allocator
corpus: consumer-side temporal defects whose stale pointer is memory the
AVBufferPool or AVRefStructPool handed out. Assembled 2026-09-19 against
`ports/ffmpeg/buffer-pool/upstream.json` = 9.0.1, using a full clone of
`git.ffmpeg.org/ffmpeg.git` at master `bfac54a03b`.*

**Result: of twenty candidates, three are pool-backed. None is live at the
current 9.0.1 pin; all three are live at n8.1, which is the newest tag that has
them and still has the public `libavutil/refstruct.c` the port extracts.**

## The blindness argument is the strongest of the three ports

Verified in the pinned source, not assumed:

- **`pool_release_buffer` returns a payload to a LIFO freelist, never to
  `malloc`** — `libavutil/buffer.c:344-352` pushes onto `pool->pool`;
  `av_buffer_pool_get:390-411` pops the same entry and rebuilds a reference over
  the identical `buf->data`. Same-address reuse is a property of the allocator,
  not of a run, as with PostgreSQL's Slab.
- **`av_refstruct_pool` is the same shape** — `libavutil/refstruct.c:223-231`
  pushes onto `pool->available_entries`, `:250-261` pops it.
- **Every decoded frame's planes are pool memory on the default path** —
  `libavcodec/get_buffer.c:196` and `:233` set
  `frame->buf[i] = av_buffer_pool_get(pool->pools[i])` inside
  `avcodec_default_get_buffer2`. CPython hides one allocator's blocks;
  FFmpeg hides the entire decoded-frame working set.

## The first instrument was wrong, and by two orders of magnitude

The first pass searched the files that **call** the pool API — 45 files. Those
are the pool's *managers*. The consumers are the 696 files under `libavcodec`,
`libavfilter` and `libswscale` that receive an `AVFrame` and use `frame->data[]`,
because the planes are pooled no matter who allocated them. On the manager set
the temporal-fix population reads as 11 over twenty years; on the consumer set it
is **278** (libavformat 114, libavcodec 112, libavfilter 48, libswscale 4). The
first number was recorded as thin population and it measured the file list.

Two weaker instruments were also discarded, and are recorded so they are not
tried again:

| instrument | reading | why it is wrong |
|---|---|---|
| message keywords only | 337 temporal fixes in 25 years | FFmpeg's convention is a `Fixes: …clusterfuzz…` trailer — 2,993 commits carry one, 9 of which say anything temporal |
| `-G` pickaxe on the fix diff | 16 frame-lifetime fixes | a lifetime fix often changes a loop bound or an order, touching none of the ref/unref tokens; it lost `af_join` and `af_adeclick` |

The population is genuinely spatial-dominated — "overflow" 2,415, "out of array"
1,062, against 0 of 115 CVE-carrying commits saying anything temporal — but the
usable instrument is a shape search over the consumer surface, cross-checked per
candidate at the source.

## Candidates and verdicts

Twenty consumer temporal fixes land after n8.0. Liveness is mechanical: the
fix's removed lines are searched verbatim in the same file at the tag, so a
defect introduced after the tag cannot be counted. Memory origin is read from
the allocation site, never inferred from the subsystem.

| Fix | Consumer | Stale object | Origin | Verdict |
|---|---|---|---|---|
| `461fb22053` | `avfilter/af_join` | `frame->extended_data[i]` | `av_frame_get_plane_buffer` on the input frame | **POOL** |
| `1886c3269d` | `avcodec/h264_refs` | `sl->ref_list[][].data[3]`, `.parent` | points into `frame->data[]`; `H264Picture` | **POOL** |
| `316531e61c` | `avfilter/vidstabtransform` | `td->src` inside libvidstab state | shallow copy of the source frame's planes | **POOL** |
| `bc46eab87c` | `avcodec/vvc/thread` | `fc->ft` | `av_mallocz` | not pool |
| `f435ce22e1` | `avcodec/h2645_sei` | `dst_env` | `av_ambient_viewing_environment_alloc` | not pool |
| `2f60af465a` | `avcodec/rasc` | `s->delta` | `av_fast_padded_malloc` | not pool |
| `d6458f6a8b` | `avcodec/aacdec` USAC | `ac->tag_che_map[][]` | `av_freep(&ac->che[type][id])` | not pool |
| `a795ca89fa` | `avcodec/qdm2` | packet-list pointers | previous `AVPacket` payload | not pool |
| `ebb6ac1bc7` | `avcodec/aom_film_grain` | `s->sets[set_idx]` | film-grain side data | not pool |
| `4ede75b5f4` | `swscale/graph` | `SwsContext`/pass | `av_malloc` | not pool |
| `073c44b8bc`, `9efca1d946`, `c51789b052` | `avfilter/vf_drawtext` | cached `FT_Glyph` | FreeType, in an `AVTreeNode` | not pool |
| `b8d5f65b9e` | `avfilter/dnn_backend_tf` | TF model handle | `av_malloc` | not pool |
| `23aea13745` | `avcodec/videotoolboxenc` | strings | platform allocator; macOS only | not pool |
| `4c6217477f` | `avcodec/nvdec` | fdd-owned context | vendor path, needs the GPU | out of reach |
| `d42cd604d0`, `4e4677bf58` | Vulkan decode paths | Vulkan objects | Vulkan, needs the GPU | out of reach |
| `4b9c4b9cfb` | `swscale/ops_dispatch` | — | file does not exist at n8.0 | not live |
| `c05fc27dd3` | `aacdec_usac` | — | already in n8.0 | not live |

Three of twenty. The ratio is the finding: FFmpeg's consumer defects are
overwhelmingly on ordinary `av_malloc` state — format lists, glyph caches,
context structs, side data — which a malloc-level tool already sees. The
nested allocator's blindness is broad in *bytes* and narrow in *known defects*.

## Liveness of the three, by tag

Removed-line match count per tag:

| Fix | n8.0 | n8.1 | n8.1.1 | n8.1.2 | n9.0.1 |
|---|---|---|---|---|---|
| `316531e61c` vidstabtransform | 26/26 | 26/26 | 26/26 | 26/26 | 5/26 |
| `1886c3269d` h264_refs | 4/4 | 4/4 | 0/4 | 0/4 | 0/4 |
| `461fb22053` af_join | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 |

**n8.1 (2026-03-16) is the newest tag with all three**, and it still ships
`libavutil/refstruct.c`, so the port's extraction and both patches carry over
unchanged. Below n8.0 the pool is `libavcodec/refstruct.c` with the private
`ff_refstruct_*` names and the extraction must be rewritten; before 2023-10-07
the second pool does not exist at all.

`af_join` and `h264_refs` need no external dependency. `vidstabtransform`
requires `--enable-libvidstab`, and its interest is exactly that the stale
pointer is parked inside a third-party library's state across frames — the
shape the CPython corpus files as "parked with no bound on reuse".

## The three shapes

- **`af_join`** — `try_push_frame` tests `j == i` instead of `j == nb_buffers`,
  so once two channels share a buffer the dedup loop stops referencing genuinely
  new ones. The buffer is released while the output frame's `extended_data[i]`
  still points into it. A one-line fix, reported by external security
  researchers.
- **`h264_refs`** — the `memset` clearing `ref_list` runs to `ref_count` rather
  than the full 32 entries, so `H264Ref` records past the count keep `data[3]`
  pointers into planes of pictures that have been returned. Reported with a
  proof-of-concept bitstream by a browser vendor.
- **`vidstabtransform`** — the separate-buffer path stores a shallow copy of the
  source frame in `td->src` without allocating; a later in-place frame skips
  `vsFrameAllocate` because `td->src` looks non-null, and writes through the
  previous frame's pointer.

## Limits

This is a source triage, not a reproduction. Each verdict names the allocation
site it was read from; none of the three has been built, run, or shown to
produce a stale access under the port's oracle, and that is the next gate. The
count of three is a count of *upstream-fixed, pool-backed consumer* defects found
by this instrument on this surface — it is not a claim that FFmpeg has only three,
and the instrument's two discarded predecessors are the reason to treat any
single number here as a floor. Re-pinning has costs this document does not
weigh: two patches to rebase, a new recording, and every existing result bundle
becoming historical rather than current.
