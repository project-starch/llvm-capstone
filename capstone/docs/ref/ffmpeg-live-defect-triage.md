# FFmpeg: triage for lifetime defects live in the 9.0.1 pin, split by allocator layer

Companion to [`ffmpeg-pool-consumer-defects.md`](ffmpeg-pool-consumer-defects.md), which surveys
*pool-consumer* defects across upstream history and is the `inventory` named in
`bug-corpora/ffmpeg/pool-repros/corpus.json`. **This file asks a different question with a different
instrument** and does not replace it.

**The question, in two halves.** All three cases in `pool-repros` are `live_in_pin: false`. (a) Is
there a pool-layer defect still present in the 9.0.1 tree we compile? (b) Is there a **plain
`av_malloc`/`av_free`** defect there, which no corpus in this tree covers for any program?

Half (b) matters because every one of the ten corpora is a *nested*-allocator corpus. Plain-heap
protection is evidenced today only by our own synthetic fixtures, never by a real upstream defect.
A plain-heap case is a **control, not a new detection claim**: a free-ended use-after-free is
`1-timing` in [`sharing-bug-taxonomy-and-novelty.md`](../design/sharing-bug-taxonomy-and-novelty.md),
which CHERI's deployed revoker and ASan both catch. Its value is that it makes "the nested cases are
structurally missed" a measurement instead of an assertion.

## The instrument

| | |
|---|---|
| clone | `git.ffmpeg.org/ffmpeg.git`, full history, 127,095 commits |
| pin | tag **`n9.0.1`**, an annotated tag dated 2026-08-12 |
| population | `n9.0.1..origin/master` = **1,788 commits**. A fix here is not in our tree |
| filter 1 | lifetime wording in the subject — 15 of 1,788 survive |
| filter 2 | the **layer is read from the diff**: `nested` if it touches `av_buffer_pool` / refstruct / `av_buffer_unref`; `heap` if it touches `av_malloc`/`av_free`/`av_freep` and no pool API. Never inferred from the subsystem, per the existing inventory's own rule |

**Commit dates do not decide liveness and are not used for it.** Four of the candidates below are
dated before the pin's tag date yet are absent from the tag, because 9.0.1 was cut from a release
branch. Ancestry decides; the pinned tree proves.

## Half (b): plain-heap candidates — the control, and the strongest results here

### 1. `vvc/thread`: a free through a local alias leaves the struct's field dangling

| | |
|---|---|
| fix | `bc46eab87c4f`, 2026-09-16 — *"avcodec/vvc/thread: clear fc->ft when the frame thread is freed"* |
| also | backported as `7193b0e994`, released in **`n9.0.2`**, so it is definitively absent from our `n9.0.1` |
| consumer | `libavcodec/vvc/thread.c` |
| layer | plain heap (`av_freep`) |

The whole fix is one line: `av_freep(&ft)` → `av_freep(&fc->ft)`. `av_freep` nulls *the pointer it is
given*; given a local alias it nulls the local and leaves `fc->ft` pointing at freed storage.

**Live at the pin, read from the tree:** `n9.0.1:libavcodec/vvc/thread.c:714` is `av_freep(&ft);`,
at the end of a sequence that has already freed `ft->rows` and `ft->tasks`.

Why it is a good control case: the shape is three statements, the reduction is near-total fidelity
(the allocator is `av_malloc`/`av_free`, which *is* the real allocator here), and the stale pointer
is a struct field rather than a local, so a reader cannot object that the optimiser removed it.

### 2. `swscale/ops_dispatch`: a field read through a pointer the call already invalidated

| | |
|---|---|
| fix | `4b9c4b9cfb56`, 2026-07-13 — *"swscale/ops_dispatch: fix use-after-free when adding opaque ops passes"* |
| consumer | `libswscale/ops_dispatch.c` |
| layer | plain heap |

One line again: `(*output)->backend = comp->backend->flags;` becomes `c.backend->flags`, a local copy
taken before the call that invalidates `comp`. **Live at the pin:** `n9.0.1:libswscale/ops_dispatch.c:664`
carries the pre-fix line verbatim.

### Also in this half, not yet read

`23aea1374579` videotoolboxenc, use-after-free of strings (the platform is macOS-only, so the code
is not in our build — a reduction would still run); `c2173dcc3232` hlsenc, freed segment buffer
pointer not cleared before an error return; `4c6217477fc6` nvdec, double free on an error path.

## Half (a): pool-layer candidates — all four are one change, and all need reading

`0661ef6bb3e5` (h264), `d34492955221` (hevc), `977db8c2cbfc` (av1), `0b0870dc8cfe` (vp9), all
2026-07-16, all *"Ensure that private HW data is freed early for pictures"*. Each moves one
`av_refstruct_unref(&pic->hwaccel_picture_private)` earlier in the same function.

**Not yet classified, and the reason is worth stating.** The subject says "freed early", which reads
as a *resource-ordering* fix rather than a use-after-free, and the existing inventory records a
retraction of exactly this mistake: `a024f8c541` vp9 was filed as a pool-backed temporal specimen
and later withdrawn because *"its own fix classifies it as spatial, not temporal"*. These four must
be read against their call sites before any of them is called a temporal defect. They also touch
hardware-accelerated decode, which the app port does not build.

**So half (a) currently yields no confirmed candidate**, and the honest reading is that FFmpeg's
pool-consumer history was already exhausted by the existing inventory ("upstream history is
exhausted at four").

## Counts

| | |
|---|---|
| population | 1,788 commits after the pin |
| lifetime wording in the subject | 15 |
| plain-heap, read from the diff | 1 confirmed live + 3 unread |
| pool-layer, read from the diff | 4, all one change, all unclassified |
| confirmed live and in class, today | **2**, both plain heap |

The ratio is the point: 2 usable candidates from 1,788 commits. Expect the same elsewhere, and do
not read a small shortlist as a weak search.
