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

# Nested candidates: regions CARVED out of one allocation, 2026-10-09

FFmpeg's nested spatial row had one case (`plane-repros`). This pass mined the history for fixes whose pre-fix access left a region FFmpeg had carved out of ONE allocation by pointer arithmetic while staying inside that allocation. Every hash below was resolved and checked with `git cat-file -t`; every carve and access quote was re-read at the fix's parent before a case was built (two mismatches against the first draft were whitespace only). Twelve of the thirteen accepted are built as `bug-corpora/ffmpeg/carved-repros`; the CUDA one (`3f6bf150cb`) is excluded because its block is device memory that no arm here can run. The table and dispositions are the mining pass's, kept as found.

## Counts

| class | n |
|---|---:|
| accepted, nested (high) | 6 |
| accepted, nested (medium-high / medium) | 7 |
| **accepted total** | **13** |
| mixed: first crossing nested, a sibling crossing in the same run escapes the allocation | 1 (+3 listed unverified) |
| 2-D interleaved carve overlaps (no contiguous region to leave) — listed, **not counted** | 6 |
| rejected (opened and read) | 47 (+6 duplicates of accepted cases) |

**Shortfall, stated plainly: 13 accepted, not 15.** The bounded last sweep (4 more commits) found nothing nested.
The reason is structural and consistent with the triage doc's Shape-E finding: the large majority of upstream
"overflow / out of array" fixes either (a) leave the allocation entirely (a single-region buffer sized too small),
(b) index a *frame plane*, which on the decoder and filter paths is its own per-plane pool allocation, or
(c) cross a struct member / static table (sub-object, not carved heap). The nested shape needs a *multi-region*
scratch/table buffer **and** an access whose overshoot is smaller than the neighbouring regions — that combination is
rare. The 2-D interleaved family below is a real intra-allocation defect class but does not fit the 1-D
"leave your region" definition.

## Accepted — summary table

| # | fix (40-char) | file : function | carved block → region left → region landed in | R/W | conf |
|---|---|---|---|---|---|
| 1 | `85407c7e63722a2d723257e8cf5f281a8c9f34a4` | libavcodec/mpegvideo_motion.c : mpeg_motion_internal / qpel_motion | `sc.edge_emu_buffer` → `ubuf` (9 rows) → `vbuf` row 0 | W (then R) | high |
| 2 | `699341d647f7af785fb8ceed67604467b0b9ab12` | libavcodec/apedec.c : decode_array_0000 | `decoded_buffer` → `decoded[0]` → `decoded[1]` | W | high |
| 3 | `cd7524fdd13dc8d0cf22e2cfd8300a245542b13a` | libavcodec/apedec.c : long_filter_high_3800 | `decoded_buffer` → `decoded[0]` → `decoded[1]` | R (inert) | medium |
| 4 | `55937bb4a7df157fb08f79e7e623a16280533275` | libavcodec/alsdec.c : decode_var_block_data | `raw_buffer` → channel c samples → channel c+1 carry-over prefix | W (RMW) | medium |
| 5 | `cd0928492410c5a93959d664362cd0d0ee50b961` | libavcodec/alsdec.c : read_channel_data | `chan_data_buffer` → `chan_data[c]` (1 slot) → `chan_data[c+1..]` | W | high |
| 6 | `9d3032b960ae03066c008d6e6774f68b17a1d69d` | libavcodec/alsdec.c : read_var_block_data | `quant_cof_buffer` → `quant_cof[c]` (max_order) → `quant_cof[c+1]` | W | medium |
| 7 | `2d0bea4719588aa9caa3f452596b9748ba13059e` | libavcodec/vp9.c : update_size / vp9_decode_frame | one `av_malloc` carved by `assign()` → `above_uv_nnz_ctx[0]` → `above_uv_nnz_ctx[1]` (and [1] → `above_segpred_ctx`) | W | high |
| 8 | `2563a33856eb597c9d53b4c7cab07b6f18417740` | libavcodec/vp9.c : vp9_decode_update_thread_context / update_size | same `assign()` block → `intra_pred_data[0]` (bpp=1 size) → `intra_pred_data[1]`… | W | medium-high |
| 9 | `b5ff61695f2c0493036b463666c5936ee10da344` | libswscale/utils.c : sws_init_context (users: hScale) | `chrUPixBuf[i]` line malloc → U half → V half (`chrVPixBuf[i]`) | W | high |
| 10 | `043bcdcdb00ebcbae80d7a9f78b763b33b9f0d15` | libavcodec/svq1enc.c : svq1_encode_plane | `scratchbuf` → `temp` (rows 0..15) → `src` (rows 16..31) | W | medium-high |
| 11 | `d2213b649383bc4d9428af42943457f782493be0` | libavcodec/rv34.c : rv34_decode_slice / rv34_mc | `tmp_b_block_base` (carved at old linesize) → `tmp_b_block_y[0]` → `tmp_b_block_y[1]` / uv | W (then R) | high |
| 12 | `68226ed9ecef675895dc55a0c58d587014639a0e` | libavcodec/vorbisdec.c : vorbis_parse_setup_hdr_residues / residue decode | `channel_residues` → channel j slice (vlen) → channel j+1 slice | W (+=) | medium-high |
| 13 | `3f6bf150cb018334809bec029325b28cff8a5a9a` | libavutil/hwcontext_cuda.c : cuda_get_buffer | pool buffer, planes carved → V plane (`data[2]`, floor(h/2) rows) → U plane row 0 | W | medium |

Duplicates / chains (same defect, other hash in history): `295e05a762332c5edcc84c325e94457815a51b5c` = #2 (Libav
copy); `18f94df8af04f2c02a25a7dec512289feff6517f` (first fix, ASan heap-oob testcase) → reverted by
`e510a8251b3fc385dfe2f0482ece5643c7d66f06` → re-fixed by #4; `9853e41aa0a6cfff629ff7009685eb8bf8d64e7f` = #6
(clamp variant); `a5dc990a4eec13320e97f287640138e549d99d88` = #11; `e6d527ff729e42d80e4756cab779ff4ad693631b` =
#12; `c5fc8ae12622a507d7b9ee30ddcd3734e6de6b1d` *introduced* #1 (the 85407c7e63 message says "Regression since
c5fc8ae1"); `98d3f2359853f1908092b6244f429ced838f493b` is an earlier, incomplete change on the very line #13 fixes,
and `ffe0104574c4ebe88a0d6fdb4fb58489ce6565c5` (3 days before #13) is what made the #13 overlap live in the copy path.

Padding question (per candidate): none of the 13 crosses a *frame plane* of `av_frame_get_buffer`; all are
scratch/table/channel buffers carved by the codec, so the 32-row plane padding question is **n/a**. In each case the
landing zone is a *semantically live neighbour region*, not allocator-reserved slack (#13: the landing row is U's
logical row 0).

---

## 1. `85407c7e63` — avcodec/mpegvideo: Fix edge emu buffer overlap with interlaced mpeg4

- **File/function:** `libavcodec/mpegvideo_motion.c` `mpeg_motion_internal()` (also `qpel_motion()`, and the lowres
  copy in `mpegvideo.c`).
- **Carving (parent):** `libavcodec/mpegpicture.c:59` `int alloc_size = FFALIGN(FFABS(linesize) + 64, 32);`,
  `:79` `FF_ALLOCZ_ARRAY_OR_GOTO(avctx, sc->edge_emu_buffer, alloc_size, 4 * 68, fail);` — one buffer. In
  `mpegvideo_motion.c:321-340` it is carved into luma rows 0..17, then
  `:328 uint8_t *ubuf = s->sc.edge_emu_buffer + 18 * s->linesize;`
  `:329 uint8_t *vbuf = ubuf + 9 * s->uvlinesize;` — U gets exactly 9 rows.
- **Crossing:** `:331-335` `emulated_edge_mc(ubuf, ptr_cb, s->uvlinesize, s->uvlinesize, 9, 9 + field_based, …)`
  writes `9 + field_based` rows. With interlaced (field_based = 1) MPEG-4 the 10th row lands at
  `ubuf + 9*uvlinesize` = **`vbuf` row 0** (9 bytes wide). The V emulation (`:336`) then overwrites it, and the field
  MC that reads U at stride `2*uvlinesize` from row `field_select` reads row 9 → V data. Write crossing of one row
  (9 bytes), then a read of the clobbered row.
- **Why nested:** both regions are inside `edge_emu_buffer` (272·alloc_size bytes; U/V end ≈ 18·ls + 10·uvls).
- **Fix:** `-            uint8_t *vbuf = ubuf + 9 * s->uvlinesize;` → `+            uint8_t *vbuf = ubuf + 10 * s->uvlinesize;`
  (plus `4 * 68` → `4 * 70` and the encoder `ebuf` offset 36 → 38 rows).
- **Reduction:** one buffer of `(18*ls + 2*10*uvls)` bytes; `ubuf = buf + 18*ls`, `vbuf = ubuf + 9*uvls`; a
  10-row × 9-byte edge-emu copy into `ubuf` (stride uvls) followed by a 10-row copy into `vbuf`; assert U row 9 ≠
  expected. Fixed arm: `vbuf = ubuf + 10*uvls`. ~40 lines.
- **Confidence:** high.

## 2. `699341d647` — apedec: prevent out of array writes in decode_array_0000

- **File/function:** `libavcodec/apedec.c` `decode_array_0000()`, called from `entropy_decode_{mono,stereo}_0000()`.
- **Carving (parent):** `:1485-1486` `av_fast_malloc(&s->decoded_buffer, &s->decoded_size, 2 * FFALIGN(blockstodecode, 8) * sizeof(*s->decoded_buffer));`
  `:1490` `s->decoded[0] = s->decoded_buffer;` `:1491` `s->decoded[1] = s->decoded_buffer + FFALIGN(blockstodecode, 8);`
- **Crossing:** `:595` `for (i = 0; i < 5; i++) out[i] = …` and `:602` `for (; i < 64; i++) out[i] = get_rice_ook(…)`
  write 64 `int32` regardless of `blockstodecode`. Called with `out = ctx->decoded[0]` (`:634`, `:640`): for
  `blockstodecode < 64` the write leaves `decoded[0]` (FFALIGN(bt,8) elements) and lands in `decoded[1]`
  (overshoot `64 − FFALIGN(bt,8)` elements = up to 224 bytes).
- **Why nested — precondition:** for `32 ≤ bt < 64` the 64-element write fits inside `decoded[0]+decoded[1]`. More
  generally `av_fast_malloc` **keeps the larger buffer** from earlier full-size loops (default 4608 blocks), so in a
  real stream the short final loop is fully nested for both channels; a tiny *first* frame escapes the allocation.
  (`ctx->decoded[1]`'s own call `:642` is nested only under the retained-buffer precondition.)
- **Fix:** `-    for (; i < 64; i++) {` → `+    for (; i < FFMIN(blockstodecode, 64); i++) {` (and the `< 5` loop).
- **Reduction:** `buf = malloc(2*FFALIGN(bt,8)*4)` (or a larger retained size), `d0 = buf`, `d1 = buf + FFALIGN(bt,8)`;
  fill `d1` with a sentinel; write 64 values into `d0`; check `d1` sentinel. ~30 lines.
- **Confidence:** high.

## 3. `cd7524fdd1` — avcodec/apedec: Check length in long_filter_high_3800()

- **File/function:** `libavcodec/apedec.c` `long_filter_high_3800()`, called from `predictor_decode_{stereo,mono}_3800()`
  with `count` = `blockstodecode` (`ape_unpack_stereo` → `ctx->predictor_decode_stereo(ctx, count)`).
- **Carving (parent):** same as #2: `:1485-1491` (`decoded[0]`/`decoded[1]` carved from `decoded_buffer`).
- **Crossing:** `:889-897` `for (i = 0; i < order; i++) delay[i] = buffer[i];` with `order` = 16/128/256
  (`:941`, `:954`) and `length = count`. For `count < order` it reads `decoded0[count .. order-1]`, i.e. past
  `decoded[0]` (FFALIGN(count,8)) into `decoded[1]`. Read only.
- **Effect:** inert — when `order >= length` the following loop `for (i = order; i < length; …)` does not run, so
  `delay[]` is never used. Real crossing, no output damage.
- **Why nested:** same retained-buffer precondition as #2; with `order ≤ 2*FFALIGN(count,8)` it is nested even in a
  fresh buffer. The fix's own testcase is an ASan heap-oob, i.e. in that file the read escaped (decoded[1] call or a
  fresh small buffer).
- **Fix:** `+    if (order >= length)` / `+        return;`
- **Reduction:** as #2; read `order` elements from `d0` with `count < order`; detect that the read touched `d1`. ~30 lines.
- **Confidence:** medium (crossing certain; effect inert, so only an instrumented/bounded read shows it).

## 4. `55937bb4a7` — libavcodec/als: fix address sanitization error in decoder

- **File/function:** `libavcodec/alsdec.c` `decode_var_block_data()` (random-access block path).
- **Carving (parent):** `:2056` `channel_size = sconf->frame_length + sconf->max_order;`
  `:2059` `ctx->raw_buffer = av_mallocz_array(avctx->channels * channel_size, sizeof(*ctx->raw_buffer));`
  `:2096` `ctx->raw_samples[0] = ctx->raw_buffer + sconf->max_order;`
  `:2098` `ctx->raw_samples[c] = ctx->raw_samples[c - 1] + channel_size;` — each channel = `max_order` carry-over
  prefix + `frame_length` samples.
- **Crossing:** `:923-929` `for (smp = 0; smp < opt_order; smp++) { … *raw_samples++ -= y >> 20; … }` with
  `raw_samples = bd->raw_samples` (block start). The RA block is the first block of the frame
  (`decode_blocks_ind :1081 bd.ra_block = ra_frame;` then `:1103 bd.ra_block = 0;`). For `opt_order > block_length`
  it RMW-writes `opt_order − block_length` samples past the block. When `opt_order > frame_length` (needs
  non-adaptive order with `max_order > frame_length`) the overshoot leaves channel c and lands in **channel c+1's
  `max_order` carry-over prefix** (overshoot < max_order, so it never passes that prefix).
- **Why nested:** for c < channels−1 the landing zone is inside `raw_buffer`; for the last channel it escapes. The
  first fix's testcase (18f94df8af, "asan_heap-oob … als_05_2ch48k16b") is a stereo file whose *last* channel
  escaped — the same stream therefore crossed nested on channel 0 first. Nested-only needs channel c+1 not to trigger
  (e.g. a constant block).
- **Fix:** `-        for (smp = 0; smp < opt_order; smp++) {` → `+        for (smp = 0; smp < FFMIN(opt_order, block_length); smp++) {`
- **Reduction:** `buf[ch*(fl+mo)]`, `rs[c] = buf + mo + c*(fl+mo)`; run the RA loop on channel 0 with
  `block_length = fl < opt_order ≤ mo`; check channel 1's prefix. ~45 lines.
- **Confidence:** medium (crossing computed from parent code; the nested configuration is legal but contrived).

## 5. `cd09284924` — Fix wrong buffer allocation for MCC in ALS.

- **File/function:** `libavcodec/alsdec.c` `read_channel_data()` (multi-channel correlation).
- **Carving (parent):** `:1565-1566` `ctx->chan_data_buffer = av_malloc(sizeof(*ctx->chan_data_buffer) * num_buffers);`
  `:1579` `ctx->chan_data[c] = ctx->chan_data_buffer + c;` — **one** `ALSChannelData` slot per channel.
- **Crossing:** `read_channel_data()` treats `cd = chan_data[c]` as an array:
  `while (entries < channels && !(current->stop_flag = get_bits1(gb))) { … current++; entries++; }` — it writes
  `cd[0..entries]` (the terminating `stop_flag` too). Any channel with ≥1 dependency entry writes `cd[1]` =
  `chan_data_buffer[c+1]` = **channel c+1's slot**. Overshoot = `entries` × `sizeof(ALSChannelData)`.
- **Why nested:** for channel 0, `entries ≤ channels−1`, so the highest slot written is `chan_data_buffer[channels−1]`
  — always inside the block. Channels ≥1 can escape.
- **Fix:** `-            ctx->chan_data[c] = ctx->chan_data_buffer + c;` → `+            ctx->chan_data[c] = ctx->chan_data_buffer + c * num_buffers;`
  (with the allocation `num_buffers` → `num_buffers * num_buffers`).
- **Reduction:** array of N structs; `cd = base + 0`; loop writing `cd[k]` for k ≤ entries; check slot 1 overwritten.
  ~35 lines.
- **Confidence:** high.

## 6. `9d3032b960` — alsdec: check opt_order.

- **File/function:** `libavcodec/alsdec.c` `read_var_block_data()`.
- **Carving (parent):** `:1632-1633` `ctx->quant_cof_buffer = av_malloc(sizeof(*ctx->quant_cof_buffer) * num_buffers * sconf->max_order);`
  `:1648` `ctx->quant_cof[c] = ctx->quant_cof_buffer + c * sconf->max_order;` (`:1649` same for `lpc_cof`).
  `:1628` `num_buffers = sconf->mc_coding ? avctx->channels : 1;`
- **Crossing:** `:663-665` `opt_order_length = av_ceil_log2(av_clip((bd->block_length >> 3) - 1, 2, sconf->max_order + 1)); *bd->opt_order = get_bits(gb, opt_order_length);`
  — `opt_order` can reach `2^L − 1 > max_order` (e.g. max_order 20 → up to 31). `:686-720` then write
  `quant_cof[k]` for `k < opt_order` (and `parcor_to_lpc` writes `lpc_cof[k]`), i.e. up to `max_order+1` elements
  past channel c's slice into channel c+1's slice.
- **Why nested — precondition:** only in MCC mode, where `bd.quant_cof = ctx->quant_cof[c]` (`:1374-1375`,
  `:1397-1398`). Non-MCC uses `quant_cof[0]` with a one-slice buffer, so the overshoot escapes.
- **Fix:** `+            if (*bd->opt_order > sconf->max_order) {` … `+                return -1;`
- **Reduction:** `buf[ch*mo]`, `q[c] = buf + c*mo`; write `opt_order = 2^ceil_log2(mo+1)−1` coefficients into `q[0]`;
  check `q[1]`. ~30 lines.
- **Confidence:** medium.

## 7. `2d0bea4719` — vp9: increase buffer sizes for non-420 chroma subsamplings.

- **File/function:** `libavcodec/vp9.c` `update_size()` (carve) and `vp9_decode_frame()` (access).
- **Carving (parent):** `:327` `#define assign(var, type, n) var = (type) p; p += s->sb_cols * (n) * sizeof(*var)`
  `:329` `p = av_malloc(s->sb_cols * (240 + sizeof(*s->lflvl) + 16 * sizeof(*s->above_mv_ctx)));`
  `:341` `assign(s->above_uv_nnz_ctx[0], uint8_t *, 8);` `:342` `assign(s->above_uv_nnz_ctx[1], uint8_t *, 8);`
  `:343` `assign(s->above_segpred_ctx, uint8_t *, 8);` — 8·sb_cols bytes each, sized for 4:2:0.
- **Crossing:** `:3877` `memset(s->above_uv_nnz_ctx[0], 0, s->sb_cols * 16 >> s->ss_h);` `:3878` same for `[1]`.
  For 4:4:4/4:4:0 (`ss_h = 0`, read from the bitstream at `:491`) each memset writes 16·sb_cols bytes into an
  8·sb_cols region: `[0]` overruns by 8·sb_cols into `[1]`, `[1]` into `above_segpred_ctx`.
- **Damage:** none observable — the bytes are zeros over regions that are themselves zeroed next (`[1]` by `:3878`,
  segpred by the following memset). The crossing is real; the corruption is not.
- **Why nested:** the landing regions are later `assign()` slices of the same `av_malloc`.
- **Reachability checked at the parent:** profile 1 is accepted (only `s->profile > 1` is rejected, `:529-530`) and
  `read_colorspace_details()` returns `YUV444P/422P/440P` for profile 1 (`:485-499`), so non-4:2:0 frames do reach
  `:3877`. Caveat: this commit is the first of a same-day 4-commit series completing non-4:2:0 decoding (next:
  `6019002f0f`, `ed3e0cc715`, `d2aa6f65db`), so at the parent such streams also decode *incorrectly*; the crossing is
  reachable regardless.
- **Fix:** `-    assign(s->above_uv_nnz_ctx[0], uint8_t *, 8);` → `+    assign(s->above_uv_nnz_ctx[0], uint8_t *, 16);`
  (same commit also widens `intra_pred_data[1..2]` 32→64 and re-carves `uvblock`/`uveob` by `chroma_blocks`).
- **Reduction:** replicate `assign()` with `sb_cols`; memset `16*sb_cols >> ss_h` with `ss_h=0`; a bounds-narrowed
  slice faults, plain C completes. ~30 lines. Use a non-zero sentinel to make the crossing visible.
- **Confidence:** high (crossing), damage nil.

## 8. `2563a33856` — vp9: re-initialize internal buffers on bpp change also.

- **File/function:** `libavcodec/vp9.c` `vp9_decode_update_thread_context()` / `update_size()`; access in
  `vp9_decode_frame()`.
- **Carving (parent):** `:316` `int bytesperpixel = s->bytesperpixel;`, `:339-341`
  `assign(s->intra_pred_data[0..2], uint8_t *, 64 * bytesperpixel);` — sized for the bpp at allocation time.
- **Mechanism (verified):** frame threads copy `pix_fmt` into the next thread's `AVCodecContext`
  (`libavcodec/pthread_frame.c:200` `dst->pix_fmt = src->pix_fmt;` at the parent), so that thread's `update_size()`
  early-returns at `:320` (`… && ctx->pix_fmt == fmt) return 0;`). `vp9_decode_update_thread_context()` frees buffers
  only on `cols/rows` change (`:4321-4323`) but copies `s->bytesperpixel = ssrc->bytesperpixel` (`:4351`). So after
  an 8→10-bit switch a thread runs with `bytesperpixel = 2` on regions carved for 1.
- **Crossing:** `:4207-4209` `memcpy(s->intra_pred_data[0], …, 8 * s->cols * bytesperpixel);` writes up to
  128·sb_cols bytes into a 64·sb_cols region → into `intra_pred_data[1]`; `[1]` → `[2]`; `[2]` → `above_y_nnz_ctx`…
- **Why nested:** the block is `sb_cols·(128 + 192·bpp + lflvl + 16·mv)` bytes; overshoots end at 256·sb_cols, far
  inside it.
- **Fix:** `+         s->rows != ssrc->rows || s->bpp != ssrc->bpp)) {` (free on bpp change).
- **Reduction:** carve with `bpp=1`, then copy rows with `bpp=2`; no threading needed in the model. ~35 lines.
- **Confidence:** medium-high (trigger needs frame threading and a mid-stream bit-depth change; mechanism read from
  source, not executed).

## 9. `b5ff61695f` — sws: fix uv overwrite in 32bt

- **File/function:** `libswscale/utils.c` `sws_init_context()` (carve); writers are the horizontal chroma scalers
  into `chrUPixBuf[i]` / `chrVPixBuf[i]`.
- **Carving (parent):** `:789` `int dst_stride = FFALIGN(dstW * sizeof(int16_t)+66, 16), dst_stride_px = dst_stride >> 1;`
  `:889-890` `if (c->scalingBpp == 16) dst_stride <<= 1;` (after `dst_stride_px` was taken)
  `:1053` `FF_ALLOC_OR_GOTO(c, c->chrUPixBuf[i+c->vChrBufSize], dst_stride*2+1, fail);`
  `:1055` `c->chrVPixBuf[i] = … = c->chrUPixBuf[i] + dst_stride_px;` — V placed at the *pre-doubling* half stride.
- **Crossing:** with 16-bit scaling the U line holds `chrDstW` `int32` = 4·chrDstW bytes, but V starts only
  `S0 = FFALIGN(2*dstW+66,16)` bytes in. U's write crosses into V by `4*chrDstW − S0` bytes (V's own write then
  clobbers U's tail). Happens only when `4*chrDstW > S0`, i.e. chroma not horizontally subsampled (444-class
  output); 420/422 do not cross.
- **Why nested:** each `chrUPixBuf[i]` line is one malloc of `4*S0+1` bytes; V's end `S0 + 4*chrDstW ≤ S0 + 4*dstW < 4*S0`.
- **Fix:** `-        c->chrVPixBuf[i] = c->chrVPixBuf[i+c->vChrBufSize] = c->chrUPixBuf[i] + dst_stride_px;` →
  `+ … = c->chrUPixBuf[i] + (dst_stride>>1);`
- **Reduction:** compute S0, double it, `malloc(2*S0_doubled+1)`, `V = U + S0/2` int16 elements; write `dstW` int32
  into U and V; compare. ~30 lines.
- **Confidence:** high.

## 10. `043bcdcdb0` — avcodec/svq1enc: fix encoding of small widths

- **File/function:** `libavcodec/svq1enc.c` `svq1_encode_plane()`.
- **Carving (parent):** `:589` `s->scratchbuf = av_malloc(s->current_picture->linesize[0] * 16 * 2);`
  `:369` `uint8_t *temp = s->scratchbuf;` (rows 0..15) and `:255` `uint8_t *src = s->scratchbuf + stride * 16;`
  (rows 16..31). `temp` is further split into `temp` (cols 0..15, intra recon) and `temp + 16` (cols 16..31, inter
  prediction).
- **Crossing:** `:430` `s->hdsp.put_pixels_tab[0][dxy](temp + 16, ref + …, stride, 16);` writes 16 rows × 16 bytes at
  `temp + 16`; the last byte is at `15*stride + 31`. When `stride < 32` that exceeds `16*stride − 1` → the write leaves
  the `temp` region and lands in **`src` row 0** (the source block currently being encoded; `:435` then encodes against
  corrupted source). Additionally `temp+16` overlaps `temp` row r+1 (2-D overlap, see family below).
- **Why nested:** `src` is the next carve of the same `scratchbuf`.
- **Reachability:** `stride < 32` depends on the plane linesize alignment of the build/input (small widths, chroma).
- **Fix:** `-    uint8_t *src     = s->scratchbuf + stride * 16;` → `stride * 32`; `temp + 16` → `temp + 16*stride`;
  allocation `16 * 2` → `16 * 3`.
- **Reduction:** `buf = malloc(stride*32)`, `temp = buf`, `src = buf + 16*stride`, stride = 16; 16×16 block copy to
  `temp+16`; check `src` row 0. ~30 lines.
- **Confidence:** medium-high.

## 11. `d2213b6493` — rv34: Fix buffer size used for MC of B frames after a resolution change

- **File/function:** `libavcodec/rv34.c` `rv34_decode_slice()` (carve) / `rv34_mc()` + `rv4_weight()` (access).
- **Carving (parent):** `:1315` `r->tmp_b_block_base = av_malloc(s->linesize * 48);`
  `:1317` `r->tmp_b_block_y[i] = r->tmp_b_block_base + i * 16 * s->linesize;`
  `:1319-1320` `r->tmp_b_block_uv[i] = r->tmp_b_block_base + 32 * s->linesize + (i >> 1) * 8 * s->uvlinesize + (i & 1) * 16;`
- **Bug:** the re-carve guard `:1311` `if (!r->tmp_b_block_base || s->width != r->si.width || s->height != r->si.height)`
  is dead after a size change, because `:1294-1295` already set `s->width = r->si.width; s->height = r->si.height;`.
  So after a resolution increase the regions stay carved at the **old** linesize.
- **Crossing:** `:789` `Y = r->tmp_b_block_y[dir] + xoff + yoff*s->linesize;` with the **new** linesize; 16 rows at
  stride `ls_new` from `y[0]` end at `15*ls_new + 15 ≥ 16*ls_old` → into `tmp_b_block_y[1]`; `y[1]` writes into the
  uv area; `rv4_weight()` (`:821-834`) then reads the mixed regions. For `ls_new` up to ≈2·`ls_old` the whole
  overshoot stays inside the `48*ls_old` block; larger jumps escape.
- **Fix:** `+            av_freep(&r->tmp_b_block_base);` in the size-change branch and
  `-        if (!r->tmp_b_block_base || s->width != r->si.width || s->height != r->si.height) {` →
  `+        if (!r->tmp_b_block_base) {`
- **Reduction:** carve with `ls_old`, then write a 16-row block with `ls_new = 1.5*ls_old` into `y[0]`; check `y[1]`.
  ~35 lines.
- **Confidence:** high (dead guard is provable from the parent lines).

## 12. `68226ed9ec` — vorbis: Fix decoder bug.  (CVE-2011-3895, part 1)

- **File/function:** `libavcodec/vorbisdec.c` `vorbis_parse_setup_hdr_residues()` (bound) /
  `vorbis_residue_decode_internal()` (write).
- **Carving (parent):** `:952` `vc->channel_residues = av_malloc((vc->blocksize[1] / 2) * vc->audio_channels * sizeof(*vc->channel_residues));`
  `:1481` `float *ch_res_ptr = vc->channel_residues;` … `:1560-1562`
  `vorbis_residue_decode(vc, residue, ch, do_not_decode, ch_res_ptr, blocksize/2); ch_res_ptr += ch * blocksize / 2;`
  — per-channel slices of `vlen = blocksize/2`.
- **Crossing:** the setup check `:682` allowed `res_setup->end ≤ channels * blocksize[1] / 2` for **all** residue
  types. Types 0/1 write `vec[voffset + j*vlen + …]` (`:1353-1364`) with `voffset` up to `end−1`, so channel j's
  residue spills into channel j+1's slice by up to `(channels−1)*vlen` floats (`+=`, i.e. adds codevectors into the
  neighbour's residue).
- **Why nested — precondition:** with a 1-channel submap at `ch_res_ptr = base` the spill ends at `channels*vlen−1`,
  inside the buffer. A submap whose last channel spills escapes.
- **Fix:** `-            res_setup->end > vc->avccontext->channels * vc->blocksize[1] / 2 ||` →
  `+            res_setup->end > (res_setup->type == 2 ? vc->avccontext->channels : 1) * vc->blocksize[1] / 2 ||`
  (the same commit adds the `ch_left` submap check, which is the *escaping* half).
- **Partial / later history:** the per-channel bound is `blocksize[1]/2` while `vlen` can be `blocksize[0]/2`
  (short blocks), so the fix was incomplete; `f74ce3a60d6ef49080df85c44b54280357109f56` added a
  remaining-buffer bound (`max_output > ch_left * vlen`), and `0a266cb55af9794fc5cff695d35cae4111e4334f` (2014)
  **removed the per-channel `end` check outright**. See aside A.
- **Reduction:** `buf[2*vlen]`, decode a type-1 residue for channel 0 with `end = 2*vlen`; check channel 1's slice.
  ~35 lines.
- **Confidence:** medium-high.

## 13. `3f6bf150cb` — avutil/hwcontext_cuda: fix yuv420p V/U plane overlap in cuda_get_buffer()

- **File/function:** `libavutil/hwcontext_cuda.c` `cuda_get_buffer()` (carve); `cuda_transfer_data()` (access).
- **Carving (parent):** `:180` `int size = av_image_get_buffer_size(ctx->sw_format, ctx->width, ctx->height, priv->tex_alignment);`
  `:185` `av_buffer_pool_init2(size, …)`, `:198` `frame->buf[0] = av_buffer_pool_get(ctx->pool);`,
  `:202` `av_image_fill_arrays(frame->data, …, frame->buf[0]->data, …)` — one buffer — then the YUV420P re-carve
  `:210` `frame->linesize[1] = frame->linesize[2] = frame->linesize[0] / 2;`
  `:211` `frame->data[2] = frame->data[1];`
  `:212` `frame->data[1] = frame->data[2] + frame->linesize[2] * (ctx->height / 2);` — V gets floor(h/2) rows.
- **Crossing:** `:264` copies `.Height = AV_CEIL_RSHIFT(src->height, …)` rows per chroma plane. For odd height V's
  last row (index floor(h/2)) is at `data[2] + ls2*floor(h/2)` = **U row 0**; whichever plane is copied last wins.
  One row (`WidthInBytes` = min linesize) of overlap.
- **Why nested:** both chroma planes are carved inside one pool buffer of `av_image_get_buffer_size` bytes.
- **Fix:** `-        frame->data[1] = frame->data[2] + frame->linesize[2] * (ctx->height / 2);` →
  `+ … * AV_CEIL_RSHIFT(ctx->height, 1);`
- **History:** `98d3f23598` changed this same line from `ls*h/2` (mid-row) to `ls*(h/2)` — overlap kept. Before
  `ffe0104574` the transfer copied only floor(h/2) chroma rows (`:264` at that parent:
  `.Height = src->height >> …`), so the overlap was latent in the copy path until then.
- **Medium because:** upstream's access is `cuMemcpy2D` on **device** memory; the reduction is a host-memory model
  of the carve, which is faithful to the arithmetic but not to the access instruction.
- **Reduction:** `buf = malloc(Y + 2*ls2*ceil(h/2))`, carve as above with odd h, copy ceil(h/2) rows into V then
  check U row 0. ~30 lines.

---

## Mixed — nested first crossing, escaping sibling crossing in the same run (not counted)

| fix | what | why mixed |
|---|---|---|
| `3b9e6a7333c1c8334e338394851e2e93f949e9e6` avcodec/magicyuvenc: fix correlation buffers size when slices are used | parent `:200` `decorrelate_buf[0] = av_calloc(2U * avctx->height, FFALIGN(avctx->width, av_cpu_max_align()));` `:203` `decorrelate_buf[1] = decorrelate_buf[0] + avctx->height * FFALIGN(…)`; `predict_slice()` `:497` `for (int i = 0; i < slice_height; i++)` runs full `slice_height` on the last slice, so `nb_slices*slice_height − height` rows overflow. | In the **same iteration** the `decorrelated[0]` row lands in `buf[1]` (nested) and the `decorrelated[1]` row lands past `2*height` rows (escapes). A per-allocation bound catches it one `diff_bytes` later, so it does not discriminate. E.g. height 10, 4 slices → slice_height 3 → 2 rows over. |

Listed for follow-up, **not verified at the parent** (do not cite as nested without reading them):
`52da3f6f70b1e95589a152aaf224811756fb9665` (EXR uint32 channel counted as 1 byte → per-channel offsets in
`uncompressed_data` wrong; within-line crossing plus an end-of-buffer escape),
`af63ea7078c8e43bc9299acbe2758b21623cffc4` (VP9 1-pass `block_base` carve used with 2-pass indexing; crossing into
`uvblock`/`eob` carves then escapes after ~1.5 superblocks), `cd1b7e2bd758165127106769a588a6384e41e9aa` (VP9
pix_fmt change under frame threads; `uvblock` carve sized for 4:2:0 used for 4:4:4), and
`32304f6cb481e216f1a941aadd6c8d1c2e7a27df` (2026 swscale ops tail buffer: one `av_fast_mallocz` carved into
per-plane `tail->in[i]`/`tail->out[i]` regions, parent `ops_dispatch.c:343-356`; `tail.in[i] += y * tail.in_stride[i]`
at `:438` indexes by output row — UNRESOLVED whether the over-read stays in the next region).

## 2-D interleaved carve overlaps (real intra-allocation defects; NOT counted)

These carve two blocks **side by side in the same rows** of one scratch buffer (e.g. U at columns 0..8, V at column
16) and rely on the stride being wide enough. When the stride is small, row r+1 of one block lands on row r of the
other. As 1-D byte ranges the two footprints overlap for **any** stride, so there is no contiguous region to "leave":
a bound-narrowing tool has nothing to narrow to. They are a distinct cell — intra-allocation, but not expressible as
per-region [base, len).

| fix | file | layout |
|---|---|---|
| `c5fc8ae12622a507d7b9ee30ddcd3734e6de6b1d` avcodec/mpegvideo: fix edge emulation with uvlinesize below 25 | mpegvideo_motion.c | `uvbuf` / `uvbuf + 16`, 9×9 blocks; collide when uvlinesize < 25 |
| `4a0ec85b85090c5ed7af232007d3e11308c926dc` avcodec/rv34: fix edge emu with uv stride <= 25 | rv34.c | same shape |
| `759e793823526e9bb72f5266d17cf6e19a33dcf3` avcodec/vc1dec: Fix support for small widths/linesizes | vc1dec.c | same shape |
| `74fd2c3ddbaf1fef5c4777784aa72b5747ad389c` avcodec/h264_mb: Fix tmp_cr for arm | h264_mb.c | `bipred_scratchpad`, `tmp_cr = +16<<ps`; collides at STRIDE_ALIGN 16 |
| `ca9eb9305aa21c7d579b29c6499d2a50c88aab47` mpegvideo_enc: fix edge emulation of dimension%16 != 0 for YUV != 420 | mpegvideo_enc.c | Cb at `ebuf+18*wrap_y`, Cr at `+8`; Cb is 16 wide for 4:4:4. **The one case where the crossing is contiguous within a row** (Cb's 16-byte row segment runs over Cr's first 8 bytes). |
| `504475f38ef049996c2c7954de00e92668a05de5` avcodec/mpegvideo: dont overwrite emu_edge buffer | mpegvideo_enc.c | encoder `ebuf = edge_emu_buffer + 32` shares rows with the MC edge-emu area |

---

## Aside A, resolved: case 11 (vorbis) is LIVE at n9.0.1 by source reading

The lead this pass left open was read to the end and adversarially audited (claim-auditor, 2026-10-09). The per-type end check `68226ed9ec` added was removed by `0a266cb55a` (2014) and never re-added (`git log -G'(vr|res_setup)->end' 0a266cb55a..n9.0.1 -- libavcodec/vorbisdec.c` is empty; the same query over all history lists both). At the pin the only bound on a type-0/1 residue is the remaining buffer, `max_output > ch_left * vlen` (n9.0.1 `libavcodec/vorbisdec.c:1441`), and the Vorbis I spec section 8.6.2 DOES clamp `residue_end` to blocksize/2 for formats 0 and 1 -- so the stream is legal and a conforming decoder truncates where n9.0.1 adds into the next channel. Two refinements from the audit: the crossing needs `begin + ptns_to_read * partition_size > vlen`, not merely `end > vlen`; and the 2012 check bounded `end` by `blocksize[1]/2` while the write uses the current block's `vlen`, so short-block packets crossed even while it stood. Not demonstrated on a stream; `carved-repros/11_*/case.json` says what would settle it.

## Aside B — triage-doc open item closed

`30c6667dadb085a2c1b9a7b7fe859f758de7044c` (snowenc `get_dc()` OBMC read): the index underflow
`obmc[index - block_h*obmc_stride]` reads below a **static const** OBMC weight table (`ff_obmc_tab[...]`), i.e. a
global object, not a carved heap region → **not nested**; it should be re-filed as such.

---

## Rejected (opened and read), one line each

From the prompt's starting list:
- `73db0bf1b06084022db5f42377b3b7960b3d3f5e` mpegvideo: increase scratchpad sizes — whole `me.scratchpad` undersized; overrun leaves the allocation (not nested).
- `330deb75923675224fb9aed311d3d6ce3ec52420` mpegvideo: set correct offset for edge emulation buffer — pointer sat mid-allocation; overrun ran past the end (not nested).
- `a30a8beeb3dc44b666d0e1aefbd823752f321ac1` vp9: Fix emu[] edge overflow conditions for >8bpp/non-420 — block overhang written into the frame plane past `linesize` (row wrap inside a per-plane pool buffer; not a carved region).
- `06f5ed40f8fceb2542add052c57608121eda2f41` avcodec/snow: Fix off by 1 error in run_buffer — `run_buffer` is its own allocation (not nested).
- `d132683ddd4050d3fe103ca88c73258c3442dc34` avcodec/snowdec: Fix off by 1 error — `hcoeff[HTAPS_MAX/2]` struct member (sub-object, not carved).
- `c20f4fcb74da2d0432c7b54499bb98f48236b904` avcodec/ffv1dec: Fix out of array read in slice counting — reads below the packet buffer start (not nested).
- `bfa0f96586fe2c257cfa574ffb991da493a54da1` vp8: fix overflow in segmentation map caching — fixed-size member queue / lifetime (not a carved crossing).
- `5029a406334ad0eaf92130e23d596e405a8a5aa0` H.264: fix overreads of qscale_table — unpadded standalone table read at negative index (leaves the allocation).

Others:
- `295e05a762332c5edcc84c325e94457815a51b5c`, `18f94df8af04f2c02a25a7dec512289feff6517f`, `e510a8251b3fc385dfe2f0482ece5643c7d66f06`, `9853e41aa0a6cfff629ff7009685eb8bf8d64e7f`, `a5dc990a4eec13320e97f287640138e549d99d88`, `e6d527ff729e42d80e4756cab779ff4ad693631b` — duplicates of accepted cases (see above).
- `38230db7b908af34315cffe848a83989dbe1678e` vp9 realloc on resolution change w/o tile_cols change — the stale object is most likely `s->entries` (own allocation); UNRESOLVED, not nested as far as read.
- `4147b337c10588b36a537c15c4b0b2b432fcc3ea` vp9 memory corruption after failed header — whole-buffer staleness across threads; not shown nested.
- `b849ac006b667dbd494a28de2f8b059fec308ac2`, `d7d3f1af2ab23cae1b2fc36afafc9872bc633808` mpegvideo_dec lowres edge-emu checks — reads past a reference chroma plane (per-plane pool; not nested).
- `e32b2c8886877a18e2951d9643c07020c76f6453` vf_kerndeint min height — out-of-plane rows of filter-pool frames; the `av_image_alloc` tmp buffer is not the one crossed.
- `a528a54ee119dcba47e7c9e30d3a56206fbad416` vf_tiltandshift 422 offset — column index past chroma width in filter-pool frames (not nested).
- `ce5274c1385d55892a692998923802023526b765` vf_fieldmatch heap-buffer overflow — `tbuffer` single allocation undersized.
- `30c6667dadb085a2c1b9a7b7fe859f758de7044c` snowenc get_dc OBMC — static table (Aside B).
- `11684476268e5c518f70e9e120f4fc6ed58ff3ef` snowenc get_block_rd memcpy — negative length → huge copy (escapes).
- `d67a6d27c25ae27d1997e0954cc4aded5d599f4c` snowenc chroma edge extension — edge-margin write lands on the last image column (2-D margin of one plane).
- `02e4970bc9d3215f862a5d64ec48922d98eb17c1` vc1 overlap filter — `over_flags_plane` read below its start / row wrap of one table.
- `f27b22b4974c740f4c7b4140a793cac196179266` h264 444 border xchg — `top_borders[…][mb_x+1]` one past the whole array.
- `2c0559d5e2faeafa7998173a4dc430408475503f`, `7322483d72d4abefae9f5c08c611f521de7236a5` mpegvideo buffer sizes — whole allocation undersized.
- `099d6813c27faf95257a529aa2c65dfde816a487` svq3 thread_count 0 — zero-sized tables (escape).
- `145061a1769d5541a5f1e9efd186717eac51c75d` h264 threaded tables — threads sharing one row ring (aliasing/race, not a bounds crossing).
- `5b4da8a38a5ed211df9504c85ce401c30af86b97`, `3a9292aff320d7b5048b371b1babea2f9b3c4e69` mv_penalty — static 2-D table (global, not heap).
- `cc17277c36cd7a87ccc99ae0e1b9eb88cbaa43ca` ffv1 bayer unaligned slices — row past a single-plane frame (per-plane pool).
- `027f60f32b758aa8e7c08685729084b1a12d81e9` ffv1 sample buffer size — whole allocation undersized.
- `8652f4e7a15e56fadf9697188c1ed42c9981db82` iff ham_palbuf — whole allocation over-read.
- `def04022f4a7058f99e669bfd978d431d79aec18` zmbvenc prev offset — reads before the allocation start.
- `e42fc6263379176869dd9a6467e37f9956d56431` mobiclip left pixel — frame plane index −1 (not nested).
- `0ec7b71de82442bb4ff6398bb2c7ec7e9f6e4f57` flac `decoded[pred_order-1]` with order 0 — below a per-channel allocation (2008 layout).
- `5f5e6af16982c172997abc75ff7a401124dd3dda` resample 8-bit — whole buffer undersized.
- `e46ad30a808744ddf3855567e162292a4eaabac7` vp8 fixed-size edge emu buffer — single region, row wrap/escape.
- `4f4334bcbcf177739fc0e159683408aeed66edc5` vf_waveform typo — wrong linesize on an input frame plane, not the carved `peak` buffer.
- `7117547298b13d6f52a20d6a62a27dc0a1c3e263` hevc sao buffer size — frame from `ff_get_buffer` (per-plane pool).
- `94bb1ce882a12b6d7a1fa32715a68121b39ee838` alsdec revert_channel_correlation range — the fix bounds the **whole** `raw_buffer`, i.e. it only stops escapes (cross-channel reads inside it remain allowed by design).
- `c36fc857b5a8f8bdf2bcc54ce72bbf817902edcf` alsdec `r` check — static `ltp_gain_values` table.
- `61f70416f8542cc86c84ae6e0342ba10a35d7cba` dca_lbr freq — static `ff_dca_freq_to_sb` table / member array.
- `d85ebea3f3b68ebccfe308fa839fc30fa634e4de` ac3enc `blocks[blk1]` — struct member array (sub-object).
- `8ca9a68f1905ff871690be38348d62a25aef2a8f` flacdec rice_order — uninitialised read, not a crossing.
- `b8b36717217c6f45db71c77ad4e7c65521e7d9ff` cfhd minimum band dimension — subbands are carved from `idwt_buf`, but the fix's testcase escaped (ASan heap-oob) and the per-band overshoot was not computed; UNRESOLVED, needs the full band layout.
- `8cac86e091118dab5a7463cb6119c940a6b8940c` vorbisdec FASTDIV with ch==1 — wrong index magnitude, escapes.
- `a6af5da7a2f817d52ea00e2aa93ccf5804afa3e0` swresample last samples — over-read of the caller's input; class depends on the caller's allocator.
- `71f9ea2d840ee69482c12c79c9987a641331c1f8` aecho non-finite delay — float→int UB, magnitude escapes.
- `49c6a44c4eedfc90ca82f25d7ea7cb02085b4429` af_chorus count mismatch — separate per-parameter arrays.
- `0794494c8f2f756e3c9384dba21c54f7d4ba9286` alsdec sb_length in RA block — not analysed far enough to place the write; UNRESOLVED.

## Method (for reproducibility)

1. Read the prompt's nine starting hashes at their parents.
2. `git log -i -E --grep 'out of (array|bounds)|out-of-bounds|overread|over-read|overflow|overlap|buffer over|past the end|off[- ]by[- ](1|one)|heap-buffer|corrupt' -- libavcodec libavutil libavfilter libswscale libswresample`
   (3,553 commits) → diffs scanned for carved-pointer names (197 hits) → read by hand.
3. `git log -E -G '<ptr> = <buf|base|scratch|tmp…> + …'` with fix wording (216 commits) → read by hand.
4. A HEAD scan for "allocate, then assign `X = alloc + k`" sites (77 sites) → per-file fix histories.
5. Subject searches for "overlap/overwrite/corrupt", "(resolution|size|bpp|format) change", "next channel/plane".
Every candidate is a sample with stated reasons, not a complete enumeration.
