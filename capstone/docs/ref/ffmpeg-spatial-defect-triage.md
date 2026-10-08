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

### (a) Frame planes — ONE allocation only on the API path. **RETRACTED and corrected.**

> **RETRACTED 2026-10-05, same day.** The first version of this section said video frame planes are
> carved from one buffer and that this is *"where FFmpeg's plane-indexing overflows live"*. **The
> second half is false.** There are **three** frame allocators at this pin and only one carves:
>
> | path | allocation | class |
> |---|---|---|
> | decoder default (`avcodec_default_get_buffer2`) | **one pool per plane** — `libavcodec/get_buffer.c:122` `pool->pools[i] = av_buffer_pool_init(size[i] + 16 + STRIDE_ALIGN - 1, ...)`, taken at `:233` `pic->buf[i] = av_buffer_pool_get(pool->pools[i]);` | not C |
> | filter default (`ff_get_video_buffer`) | **one pool per plane** — `libavfilter/framepool.c:70` `pool->pools[i] = av_buffer_pool_init(sizes[i] + align, ...)` over `FFALIGN(pool->height, align)` rows | not C |
> | `av_frame_get_buffer` (the API path) | **one allocation, planes carved** — below | **C** |
>
> Decoding and filtering are where essentially all plane indexing happens, so the claim pointed at
> the wrong population. Verified by reading both files at `n9.0.1`, not taken on report.

What survives is the API path. `n9.0.1:libavutil/frame.c`, `get_video_buffer`:

```c
frame->buf[0] = av_buffer_alloc(total_size);                      /* :122 -- ONE allocation */
...
if ((ret = av_image_fill_pointers(frame->data, frame->format, padded_height,
                                  frame->buf[0]->data, frame->linesize)) < 0)   /* :128-129 */
```

so for a frame the caller allocated with `av_frame_get_buffer`, an overflow of plane Y does land in
plane U inside one allocation. **The class of a defect in encoder/filter code therefore depends on
which allocator the CALLER used**, which has to be stated per candidate rather than assumed.

### (a′) The per-plane pool slack is a blind spot for EVERY arm, and it is not a nested effect

Both pool paths deliberately over-allocate: `size[i] + 16 + STRIDE_ALIGN - 1` in the decoder,
`sizes[i] + align` over `FFALIGN(height, align)` rows in the filter. An `AVBufferPool` buffer is an
individual `av_buffer_allocz`, so **the pool chunk IS one allocation**. Therefore, for a *plane*
overflow:

- within the slack → invisible to `shrink`, to `sublet`, **and to the pool arms**, because every one
  of them bounds the chunk, and the slack is inside the chunk;
- past the chunk → caught by `shrink` already, so the pool arms add nothing.

This is consistent with what the port already measured: FFmpeg fixture 14 `pool_one_past` faults on
`poolstock` too. **So FFmpeg's plane overflows give the pool arms no spatial discrimination at all.**
The blind spot is over-allocation, not nesting — worth stating because it is easy to mistake for a
nested-allocator gap.

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

## 2. The port cannot measure any of this — and for FFmpeg the reason is sub-object granularity

Every FFmpeg port patch targets the **pool** APIs and nothing else:

- `ports/ffmpeg/sublet/ffmpeg-9.0.1-0001-avbufferpool-on-sublet.patch`
- `ports/ffmpeg/sublet/ffmpeg-9.0.1-0002-refstruct-on-sublet.patch`
- `ports/ffmpeg/buffer-pool/patches/*-pool-*.patch`

**No patch in the tree touches `av_buffer_alloc`, `frame.c` or `av_image_fill_pointers`** (checked
across every `.patch` under `ports/ffmpeg/`). So the port narrows pool payloads only.

**None of §3's four defects is therefore measurable by any arm we have**, and for a sharper reason
than the tshark `BLOCK_FAST` gap. Those four cross a bound *inside a single `av_mallocz` or
refstruct* — between two members of one struct. No allocator adapter can help: the authority that
would have to be narrowed is **per struct member**, which means the compiler or the allocation site,
not the allocator. That is the `partial²` cell of the taxonomy (*"allocation-granular, not
sub-object"*), and `table6-cheri-vs-capstone-explained.md` already gives CHERI and Capstone the same
verdict there.

So the two programs fail to measure spatial for *different* reasons, and the distinction matters:
tshark's gap is **unwired allocator coverage** and closing it is a port change; FFmpeg's is
**sub-object granularity** and closing it is not an allocator problem at all.

Note `refstruct` is a partial exception: the sublet patch does put refstruct *pool* objects under
leases, which is why fx16 discriminates on `pool0`/`pool2`/`poolsublet`. What it does not cover is a
refstruct object allocated outside a pool.

## 3. Candidates: four class-C defects, all LIVE at the pin — and THREE are now built

**Cases 0, 1 and 2 of `bug-corpora/ffmpeg/subobject-repros/`**, a new sibling corpus, measured 3/3 natively with `runners/run-native.sh` exit 0 (`results/20261005-native-subobject/`). The oracle is the upstream fix, because it is the only one available: **no configuration we have catches any of them**, and ASan is measured blind with a positive control that fires. `a809a784ec` remains unbuilt — its containment is partial, so a corpus is the wrong place for it.

Class C in FFmpeg does **not** live in frame planes (see the retraction in §1a). It lives in
**parameter-set, SEI and hwaccel-private structs**, each allocated as ONE `av_mallocz` or one
refstruct, where an array member overflows into the **next member of the same struct**. That is a
sub-object crossing no per-allocation bound can see.

### 1. `8864fd0aec` — CBS H.265 `pic_timing` SEI. Verified here line by line.

The cleanest of the set: a 2-byte member-to-member crossing, fully contained, no magnitude at which
it escapes the allocation. All quoted from `n9.0.1`:

```c
/* libavcodec/cbs_h265.h:627-628 -- adjacent members */
uint16_t num_nalus_in_du_minus1[HEVC_MAX_SLICE_SEGMENTS];
uint32_t du_cpb_removal_delay_increment_minus1[HEVC_MAX_SLICE_SEGMENTS];

/* libavcodec/cbs_h265_syntax_template.c:1997 -- bound is 600 INCLUSIVE */
ue(num_decoding_units_minus1, 0, HEVC_MAX_SLICE_SEGMENTS);
/* :2004-2005 -- so i reaches 600, one past a 600-element array */
for (i = 0; i <= current->num_decoding_units_minus1; i++) {
    ues(num_nalus_in_du_minus1[i], 0, HEVC_MAX_SLICE_SEGMENTS, 1, i);
```

`HEVC_MAX_SLICE_SEGMENTS = 600` (`libavcodec/hevc/hevc.h:150`). Index 600 of a `uint16_t[600]` sits
at byte offset 1200, which is 4-aligned and is exactly where
`du_cpb_removal_delay_increment_minus1` begins — so the write lands on that member's low half. The
struct is **one allocation**: `libavcodec/cbs_sei.c:257`
`av_refstruct_alloc_ext(desc->size, ...)` → `libavutil/refstruct.c:109` `av_malloc(size + REFCOUNT_OFFSET)`.

- **Liveness:** read from the pin above — the bound at `:1997` is still the inclusive
  `HEVC_MAX_SLICE_SEGMENTS`, and the fix's `cbs_h265_pic_size_in_ctbs(...)` call does not appear in
  this function.
- **Reachability:** `cbs_h265` is not the HEVC decoder; the entry points are the `h265_metadata` and
  `trace_headers` BSFs and the VAAPI/Vulkan H.265 encoders.

### 2. `a809a784ec` — VVC slice-header entry points

`libavcodec/vvc/ps.c` writes `sh->entry_point_start_ctu[j++] = i;` with no bound on `j`.
`entry_point_start_ctu[VVC_MAX_ENTRY_POINTS]` is the last member of `VVCSH`, and `VVCSH sh` is the
second member of `SliceContext`, allocated whole at `libavcodec/vvc/dec.c:505`
`av_mallocz(sizeof(*fc->slices[0]))`. So the first out-of-bounds words land on the following
pointer members (`eps`, `nb_eps`, `rpl`, `ref`) — **pointer corruption inside one allocation**.
Containment limit, stated honestly: roughly the first 32 bytes are intra-allocation; far past that
it leaves the `SliceContext`. The bound crossed *first* is the sub-object bound.
**Live:** the pin still has the unguarded `static void sh_entry_points(...)` and
`grep VVC_MAX_ENTRY_POINTS` over the pinned `ps.c` finds no match — the guard is absent, not moved.

### 3. `68845e26f7` — Vulkan HEVC reference-set lists. Fully contained.

`libavcodec/vulkan_hevc.c:775` writes `hp->h265pic.RefPicSetStCurrBefore[i] = j;` for
`i < h->rps[ST_CURR_BEF].nb_refs`. The arrays hold **8** entries (the pin's own
`memset(..., 0xff, 8)` at `:768` proves it) while `nb_refs` can reach `HEVC_MAX_REFS = 16`. So up to
8 bytes past an 8-byte array, into the neighbouring reference-set fields of the same `h265pic`
member, inside one refstruct allocation (`libavcodec/decode.c:2352`). **No magnitude escapes the
allocation.** Vulkan-hwaccel builds only.
**Live:** the fix's three-way `FF_ARRAY_ELEMS` guard is absent; the pin runs straight from `:767`
into the `memset` at `:768` and the unguarded loop.

### 4. `e058af88ab` — Vulkan HEVC DPB fill

`libavcodec/vulkan_hevc.c:760` fills `hp->ref_src[idx]` for `idx` up to 31, because the loop walks
`l->DPB` which is `HEVCFrame DPB[32]`, while the targets are `HEVC_MAX_REFS = 16` wide. The first
out-of-bounds write is `ref_src[16]`, landing on `h265_refs[0]` — the next member, same refstruct
allocation. **Live:** the fix's `if (nb_refs >= HEVC_MAX_REFS)` guard is absent and the call sits
unguarded at pin line 760.

### Two further candidates, kept but conditional

`ed27b2c498` (AMV chroma flip, `libavcodec/mjpegenc.c:636-637`) is class C **only when the caller
used `av_frame_get_buffer`**; from a filtergraph the underflow leaves the per-plane pool chunk and
becomes class A. It needs odd `avctx->height` and `-strict -1`.

One further candidate is an interlaced ProRes encoder reading past the bottom of the input field
(`ad2287d6d2` with companion `d74f5e0559`). **Its file is named after a contributor, so the path is
deliberately not written here; cite the commit hashes.** Two things make it interesting and one
makes it awkward: its class follows the caller's allocator as above, **and on our target the class-B
reading fails** — the filter pool pads to `FFALIGN(height, av_cpu_max_align())`, so with
`align >= 32` (x86-64) the over-read stays inside the chunk, but at `align = 16` there is **zero
slack** and it leaves the chunk entirely. Worth recording precisely because a class claim that holds
on the development host and not on the target is the kind that gets published wrongly.

## 4. The automated pass found none of these, and that is an instrument result

**`spatial-triage.py --program ffmpeg` is not the instrument that found the four above.**

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

The shape search is what produced §3's four, by enumerating the single-allocation seams from §1
and then reading each candidate's allocation site. It also caught **5 more backport traps** -- five
candidates in `n9.0.1..origin/master` whose fix is already in the pin under another hash
(`4f8043e658`, `ed4f286a10`, `c42ac5c566`, `37721094ab`, `3b939ced79`), independently reproducing
the false-live mode measured in `memcached-spatial-defect-triage.md` §5.

## What this instrument cannot see

- Everything in §3: the allocation site when it is more than one alias hop from the fix.
- A defect never fixed upstream, and a fix that does not word itself spatially.
- Liveness for all four §3 candidates was read from `n9.0.1:<path>` directly, never from filter 3,
  which has a measured ~20% false-live rate (`memcached-spatial-defect-triage.md` §5). Only
  `8864fd0aec` was re-verified line by line in this session; the other three rest on one reading
  each and should be re-read before any of them enters a case folder or the paper.
- **Reachability is not established for any of them.** Three need specific builds or BSF entry
  points, named per candidate.

## PoisonCap and CheriBSD are UNMEASURABLE on this host, checked rather than assumed

Every case in this corpus declares a `poisoncap-spatial`, `poisoncap-protected` and
`cheribsd-revocation` arm, and **none of them can be measured here.**
`ports/common/cmake/toolchains/cheribsd.cmake:4-9` requires `CHERI_SDK` and `CHERI_SYSROOT` and
`FATAL_ERROR`s without them; both are **unset** even after sourcing the project environment, and a
filesystem search finds no CHERI SDK, rootfs or PoisonCap image anywhere on this host.

Recorded as **unavailable**, not *pending*. The distinction matters: "pending" invites someone to
wait for a measurement that cannot be taken on this machine.

---

# Candidate disposition, 2026-10-06

## Correction: CheriBSD is no longer unmeasurable on this host

The section above says PoisonCap and CheriBSD "cannot be measured here" and that no SDK, rootfs or
image exists on this machine. **That was true when written and is now false for CheriBSD:** a stock
CheriBSD purecap vehicle was built on this host and used for measurements on 2026-10-05/06 (kernel
`CHERI-PURECAP-QEMU` from `releng/26.07-88f39900c329`, the `CHERIBSD_REV` in `pins.env`). The
memcached and wireshark plain-heap and `wmem` rows carry readings from it. **PoisonCap remains
unmeasurable here** — that half of the claim stands. FFmpeg's CheriBSD arms are declared predictions
because nobody has run them yet, which is a different reason from the one above and the distinction
matters: *not yet done*, not *cannot be done*.

## What the population actually is, and why no "N of M triaged" claim is made

**The funnel depends entirely on the query, so the query is stated and the number is not dressed up.**
Searching both windows for spatial wording in the five library trees:

```
git log --regexp-ignore-case --extended-regexp \
  --grep 'out of (array|bounds)|out-of-bounds|overread|over-read|overflow|buffer overflow|OOB|past the end|off-by-one|heap-buffer' \
  <range> -- libavcodec libavformat libavfilter libavutil libswscale
```

| range | commits |
|---|---:|
| `n7.0..n9.0.1` | 566 |
| `n9.0.1..master` | 158 |
| **distinct over `n7.0..master`** | **663** |

**This is NOT a triaged population and is not claimed as one.** Unlike `bug-corpora/mruby`, whose 23
cases rest on a committed matrix of 674 candidates each actually *run* on three arms, FFmpeg has no
harness that can execute an arbitrary upstream fix — so there is no equivalent of that matrix to
commit, and the mruby method does not transfer. What follows is the set that was **opened and read**,
with its disposition. Treat it as a sample with stated reasons, not as a complete enumeration: a
narrower keyword set yields a much smaller population, and a count like "14 of 78" would be an
artefact of the query rather than a fact about FFmpeg.

## Dispositions

**Built** — 15 cases, in the corpus named:

| hash | shape | corpus | case |
|---|---|---|---:|
| `8864fd0aec` | A — member to next member | `subobject-repros` | 0 |
| `68845e26f7` | A | `subobject-repros` | 1 |
| `e058af88ab` | A | `subobject-repros` | 2 |
| `89de2f0de1` | A — *read* one past | `subobject-repros` | 3 |
| `1a00ea51cb` | A — underflow below a member | `subobject-repros` | 4 |
| `d29ff88422` | A — write one past | `subobject-repros` | 5 |
| `a809a784ec` | A — unbounded walk | `subobject-repros` | 6 |
| `275e217b10` | A — copy sized by source length | `subobject-repros` | 7 |
| `fb862976df` | A — unbounded walk, guard keyed to a stale counter | `subobject-repros` | 8 |
| `ac59fc542f` | B — carved sub-slice, data-controlled index | `subobject-repros` | 9 |
| `b7946098b1` | plane/row | `plane-repros` | 0 |
| `d133b4a231` | C — past a direct `av_malloc_array` | `plain-heap-repros` | 0 |
| `bcbf3a5630` | C | `plain-heap-repros` | 1 |
| `56309e476a` | C — **below** the base | `plain-heap-repros` | 2 |
| `495b402f27` | C — allocation sized for one of four carves | `plain-heap-repros` | 3 |

**Opened, classified, not built** — each with the reason, so none of them is simply forgotten:

| hash | what it is | why not built |
|---|---|---|
| `01701bdcd5` | VVC `subpic_tiles`: `while (col_bd[*tile_x] != rx) (*tile_x)++` — an unbounded scan for an equality that may never hold | shape A and buildable; the target array's declaration was not run down, so the adjacency a case must assert is unverified. **Next candidate if more are wanted.** |
| `8b9851b005` | `aacdec_usac_mps212`: `set_idx + data_pair > MPS_MAX_PARAM_SETS` should be `>=` | the off-by-one guard and the write shown in the diff index *different* arrays; which object is actually crossed was not established. Not buildable without that. |
| `18761f9fb5` | `rtpdec_av1`: `pktpos += obu_size` where `buf_ptr` was meant | a two-variable confusion rather than a one-term reversal, so the arms would differ in more than one respect |
| `9edd06f861` | verified live at the pin in an earlier pass | opened, never built — recorded here because the earlier doc said "remains a live-at-pin candidate" and then dropped it |
| `30c6667dad` | overread of the OBMC window table | disqualified against the *nested* cell's criterion and then dropped, instead of being re-filed as not-nested. The re-filing is still open. |

## Shape E: a MEASURED EXCLUSION, not a gap

The single largest reason an earlier pass over this material yielded only four cases is a class of
candidate where **the crossing never leaves an over-allocated container**. Two absorbers account for
most of it:

- **`AV_INPUT_BUFFER_PADDING_SIZE` is 64 bytes.** A bytestream overread of a few bytes past a packet
  stays inside the allocation by construction, because FFmpeg allocates that padding precisely so
  that readers may overrun.
- **Per-plane pool slack.** `av_frame_get_buffer` pads to `FFALIGN(height, 32)`. This was *measured*,
  not assumed: the alpha plane's own allocated extent came out at **1024 against a logical 160**, and
  that measurement is what forced the retraction of the `plane-repros` discriminating-arm claim.

A shape-E candidate is therefore **not a defect this inventory can count**: there is no bound for it
to have crossed, and a case built from it would report a completion on every arm for a reason that
has nothing to do with the upstream fix. **This is a finding about FFmpeg's allocation discipline,
not a shortfall in the search** — and it is the reason the plane/row shape is **exhausted at one case
and must not be grown**.

# Candidate disposition, 2026-10-08: `01701bdcd5` reclassified, and what a case from it must assert

The "Opened, classified, not built" table above lists `01701bdcd5` as **shape A** and as the
**next candidate if more are wanted**, with the reason it was not built being that *"the target
array's declaration was not run down, so the adjacency a case must assert is unverified."*

That declaration has now been run down, and **the shape is wrong: it is class C, not A, and the
row would be NOT nested.** So a case from it belongs in `plain-heap-repros`, not
`subobject-repros`.

**The evidence, read at the 9.0.1 pin rather than inferred.** The arrays the scan walks are
pointers, not inline members:

    libavcodec/vvc/ps.h:118   uint16_t *col_bd;    ///< TileColBdVal
    libavcodec/vvc/ps.h:119   uint16_t *row_bd;    ///< TileRowBdVal

and each is its own direct allocation:

    libavcodec/vvc/ps.c:352   pps->col_bd = av_calloc(r->num_tile_columns + 1, sizeof(*pps->col_bd));
    libavcodec/vvc/ps.c:353   pps->row_bd = av_calloc(r->num_tile_rows    + 1, sizeof(*pps->row_bd));

Under this inventory's axis -- **who allocated the object** -- a direct `av_calloc` is not nested.
There is also no adjacency for a case to assert, which is why that stated blocker dissolves rather
than being satisfied: the crossing leaves the allocation entirely, so the question is the
allocation's extent, not what sits next to it inside a larger block. Collapsing "which bound is
crossed" with "who allocated" is the 2026-10-05 retraction, and calling this shape A would have
repeated it.

**The defect itself is real and small.** The fix replaces `!=` with `<` in two scans:

    while (pps->col_bd[*tile_x] != rx)   ->   while (pps->col_bd[*tile_x] < rx)

`col_bd`'s last element is set to `pps->ctb_width` (`ps.c:365`), so the scan terminates for any
`rx` at or below it. The overread needs `rx` greater than every boundary -- which the upstream
message says an illegal bitstream can construct, and which is also reachable when a horizontal
subpicture boundary is not a tile boundary.

**What a case from it must get right, and the trap waiting in it.** The CheriBSD verdict is decided
by the array's SIZE, not by the defect:

- `av_calloc(n + 1, 2)` for a minimal `n = 1` requests **4 bytes**. CheriBSD's malloc bounds to the
  allocator's **usable** size, so a 4-byte request yields a 16-byte capability and a two-byte
  overread at offset 4 is **inside** it -- the arm would complete, and the row would discriminate
  nothing.
- The size must therefore be chosen so the crossing exceeds the usable size, and the capability
  length must be **measured in the guest for that request size** rather than read off the sibling
  corpora's table: a size class is a step function, and the plain-heap corpus's own bundle records
  that non-transferability as the thing which refuted one of its catch predictions.

So: buildable, genuinely open, and it is a `plain-heap-repros` row whose size choice is part of the
claim. Recorded here rather than built, so that the next pass starts from the corrected
classification instead of the one above it.

**Citation constraint for this hash.** `01701bdcd5`'s commit message carries a `Signed-off-by:`
trailer naming an outside contributor with an email address. Cite it by **hash, subject and path
only** -- the subject line itself is clean -- and never quote the trailer. The same constraint is
recorded against a memcached hash in `memcached/plain-heap-repros/01_.../case.json`.
