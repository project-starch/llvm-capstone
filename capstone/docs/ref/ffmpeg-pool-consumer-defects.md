# FFmpeg pool-consumer defects — triage at the 9.0.1 pin

*Which upstream FFmpeg defects are usable as specimens for the nested-allocator
corpus: consumer-side temporal defects whose stale pointer is memory the
AVBufferPool or AVRefStructPool handed out. Assembled 2026-09-19 against
`ports/ffmpeg/buffer-pool/upstream.json` = 9.0.1, using a full clone of
`git.ffmpeg.org/ffmpeg.git` at master `bfac54a03b`.*

**Result: four pool-backed specimens are known. One is live at the current 9.0.1
pin; all four are live at n8.1, which is the newest tag that has them and still
ships the public `libavutil/refstruct.c` the port extracts. The limit on the
corpus is not the search window but FFmpeg's defect composition: the fourth case
was only found by searching fixes that were never backported, and widening that
search further returns hardware paths a domain cannot run.**

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

Three of twenty in this window; a fourth comes from the never-backported
window below. The ratio is the finding: FFmpeg's consumer defects are
overwhelmingly on ordinary `av_malloc` state — format lists, glyph caches,
context structs, side data — which a malloc-level tool already sees. The
nested allocator's blindness is broad in *bytes* and narrow in *known defects*.

## Liveness of the first three, by tag

Removed-line match count per tag:

| Fix | n8.0 | n8.1 | n8.1.1 | n8.1.2 | n9.0.1 |
|---|---|---|---|---|---|
| `316531e61c` vidstabtransform | 26/26 | 26/26 | 26/26 | 26/26 | 5/26 |
| `1886c3269d` h264_refs | 4/4 | 4/4 | 0/4 | 0/4 | 0/4 |
| `461fb22053` af_join | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 |

**n8.1 (2026-03-16) is the newest tag with all three** — and, as the next section
shows, with the fourth as well. It still ships `libavutil/refstruct.c`, so the
port's extraction and both patches carry over unchanged. Below n8.0 the pool is `libavcodec/refstruct.c` with the private
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

## Never-backported fixes are a second, larger pool — with worse precision

A defect fixed on master after a release branch was cut, and never cherry-picked
into it, is still present in that branch's tags. This is CPython's Group B, and
for FFmpeg it is the larger of the two windows:

| Pin | branch point | Group A (branch after tag) | Group B raw | Group B never backported |
|---|---|---|---|---|
| n9.0.1 | 2026-06-26 | 7 | 34 | 19 |
| n8.1 | 2026-03-08 | 13 | 58 | 45 |
| n8.0.3 | 2025-08-09 | 4 | 78 | 59 |
| n7.1.5 | 2024-09-24 | 4 | 95 | 81 |

Backports are cherry-picks, so membership is decided by patch id (`git cherry`),
not by reachability. Group B grows steadily as the pin ages, and this is where
the fourth specimen came from.

Its precision, however, collapses in the other direction. At the 9.0.1 pin all
19 Group B candidates are Vulkan, AMF, CUDA or VideoToolbox paths, or fixes on
memory that is not pooled — **none is usable**, because a Capstone domain has no
GPU. At n8.1, 19 of 45 are still hardware; of the 26 software ones, the swscale
op-chain work, the bitstream-filter `profile` resets and the `itut35` side-data
unrefs are ordinary `av_malloc` state, and `ff_hwaccel_frame_priv_alloc`
(`decode.c:2352-2356`) uses `av_refstruct_allocz` **without** a pool, so the four
"private HW data freed early for pictures" commits are refstruct but not pooled.

The pooled object classes at this pin are exactly four, read from their
allocation sites:

- `frame->buf[i]` planes — `av_buffer_pool_get`, `get_buffer.c:196`/`:233`
- `progress_frame_pool` — `av_refstruct_pool_alloc_ext`, `decode.c:2136`
- per-decoder refstruct pools — e.g. h264 `decode_error_flags_pool`, hevc
  `tab_mvf_pool`/`rpl_tab_pool`, `MPVPicture`
- the filter frame pool — `libavfilter/framepool.c`

Anything else is not in this corpus's class, however temporal its fix reads.

## The fourth specimen, and the tally by pin

`a024f8c541` (master, backported to release/9.0 as `c878aa71a6`) —
`vp9_decode_flush()` releases `s->s.frames[]`, `s->s.refs[]` and
`s->s.ref_frames[]` but leaves `s->next_refs[]` referenced. Under frame
threading, `vp9_decode_update_thread_context()` seeds a worker's `refs[]` from
the source worker's `next_refs[]`, so pre-flush references survive the flush and
are resurrected, and a later inter frame passes the availability check and
decodes against references that no longer exist. `ProgressFrame` internals come
from `progress_frame_pool`, so this is pool memory. The reduction has to model
the thread-context hand-off, which is a copy between two contexts rather than a
race, so it is expressible without threads — but that is an argument, not yet a
fixture.

| Specimen | class | live at n9.0.1 | live at n8.1 | needs |
|---|---|---|---|---|
| `af_join` | buffer pool | no | yes | — |
| `h264_refs` | buffer pool | no | yes | — |
| `vidstabtransform` | buffer pool | no | yes | `--enable-libvidstab` |
| `vp9` next_refs | refstruct pool | **yes** | yes | frame-threading hand-off modelled |

Two candidates remain unresolved and are recorded rather than counted:
`08597a382e` (`avcodec/mwsc`, does not dereference a missing reference frame) and
`fd3ee52fab` (`avcodec/tdsc`, unrefs the reference frame before reallocating on a
size change). The second is upstream-classified as an out-of-array access; its
cause is a retained buffer used with updated dimensions, which is a size
confusion on live memory rather than a stale pointer after release, so it may not
belong in this class at all.

## Limits

This is a source triage, not a reproduction. Each verdict names the allocation
site it was read from; none of the four has been built, run, or shown to
produce a stale access under the port's oracle, and that is the next gate. The
count of four is a count of *upstream-fixed, pool-backed consumer* defects found
by this instrument on this surface — it is not a claim that FFmpeg has only three,
and the instrument's two discarded predecessors are the reason to treat any
single number here as a floor. Re-pinning has costs this document does not
weigh: two patches to rebase, a new recording, and every existing result bundle
becoming historical rather than current.
