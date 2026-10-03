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
branch.

> ### ~~Ancestry decides; the pinned tree proves.~~ — WITHDRAWN 2026-10-03
>
> **Both halves of that sentence are wrong, and it governed every verdict in this file.**
>
> 1. **Ancestry is INSUFFICIENT.** `n9.0.1` is a release-branch tag that cherry-picks, so a fix can
>    be "not an ancestor" and still be present by content under a different sha. `4b9c4b9cfb` is
>    exactly that: absent by ancestry, present as `716d2a47c5`, whose body reads
>    *"(cherry picked from commit 4b9c4b9cfb56…)"*.
> 2. **"The pinned tree proves" was a grep for a line SHAPE**, and a line shape recurs at sites where
>    the defect does not exist. For `ops_dispatch` it matched line 664, where the object is still
>    alive — see the retraction in section 2.
>
> **The replacement test, and both of its controls must be run:**
>
>     git cat-file -t <sha>                                       # MUST print `commit` -- see below
>     git log <tag> --grep='cherry picked from commit <sha>'     # non-empty  => BACKPORTED, not live
>     git log <tag> -- <path>                                     # a same-subject commit => suspect
>
> then confirm **the free precedes the read on the same path** by reading the enclosing function, not
> by grepping for the line. Controls: the probe must fire on a known backport (`4b9c4b9cfb` →
> `716d2a47c5`) and come back empty on a bogus sha. Without the first control a probe that silently
> matches nothing reads exactly like "not backported".
>
> **Step 0 — `git cat-file -t <sha>` must print `commit` — added 2026-10-03, and it is not a
> formality.** The two controls above *cannot distinguish a nonexistent candidate from a live one*,
> because the bogus-sha control is designed to come back empty and a nonexistent sha produces exactly
> that. Worse, `git merge-base --is-ancestor <bad-sha> <tag>` exits **128** ("bad object"), and 128 is
> non-zero, so it reads as "not an ancestor" — i.e. as *live*. A candidate was reported on 2026-10-03
> with all prescribed controls run and a sha that does not exist in FFmpeg's history; every probe on it
> was vacuous. This is the CLAUDE.md rule *"treat any exit status the gate does not itself define as
> BLOCKED, not as a pass"* in a new place: `--is-ancestor` defines 0 and 1, and 128 is neither.
> Resolve the sha to an object **before** any probe, and keep a resolvable control (`5c66a3ab51` →
> `commit`) beside it so step 0 is itself two-sided.
>
> This is the "POSITIVE finding from a narrowed view" trap in CLAUDE.md, in its own right: the grep
> was written to find the pre-fix line and it found one.

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

### 2. ~~`swscale/ops_dispatch`~~ — RETRACTED 2026-10-03: FIXED at our pin, never live

| | |
|---|---|
| fix | `4b9c4b9cfb56`, 2026-07-13 — *"swscale/ops_dispatch: fix use-after-free when adding opaque ops passes"* |
| **backport** | **`716d2a47c5`, IN `n9.0.1`** — body: *"(cherry picked from commit 4b9c4b9cfb56…)"* |
| consumer | `libswscale/ops_dispatch.c` |
| layer | plain heap |
| **live in pin** | **NO** |

**The claim "live at the pin, `ops_dispatch.c:664` carries the pre-fix line verbatim" is withdrawn.**
Reading the enclosing function `compile_single` (`n9.0.1:libswscale/ops_dispatch.c:527-674`) instead
of grepping it shows two distinct `backend =` sites:

- **line 553 — the defect's site, and it is FIXED.** The `p->comp.opaque` branch takes a copy,
  `SwsCompiledOp c = *comp;`, then `av_free(p);`, then reads `c.backend->flags`. That is the
  post-fix form; the cherry-pick is applied.
- **line 664 — the site I cited, and it is not a defect.** On the fall-through path `p` is never
  freed: it is handed to `ff_sws_graph_add_pass(…, op_pass_setup, p, op_pass_free, output)`, which
  takes ownership while the object stays live. Two lines later `align_pass(…, p->pixel_bits_out)`
  reads `p` directly — which is itself proof that `p` is alive there.

So `comp->backend->flags` at 664 reads a **live** object. No free precedes it on that path.

**Consequence for the measured results.** The fixture built from this (`ffapp_safety.c` fixture 25)
remains a valid *synthetic* interior-pointer use-after-free, and the six cells measured in
`../../ports/ffmpeg/app/results/2026-10-03-qemu-upstream-defects-heap-arms/` stand as a measurement
of that fixture. What does **not** stand is the framing: fixture 25 is **modelled on** upstream
`4b9c4b9cfb`, which is **fixed at the version we compile**. It is not a reproduction of a defect
live in our pin, and must not be counted as one.

### ~~Also in this half, not yet read~~ — READ 2026-10-03, all six triaged

`23aea1374579` videotoolboxenc; `c2173dcc3232` hlsenc; `4c6217477fc6` nvdec; plus
`415c9f7b35` avutil/mem, `b8b8d43935` h274 and `d42cd604d0` vulkan_ffv1, which the earlier
wording filter had not surfaced. Every one was put through the backport probe **and** the
built-or-not question against the port's own `config.h`/`config_components.h`.

| sha | subsystem | backported into `n9.0.1`? | built by the port? | verdict |
|---|---|---|---|---|
| `c2173dcc32` | hlsenc | no | **no** — `CONFIG_HLS_MUXER 0` | **live, reducible, portable C — the only usable one** |
| `b8b8d43935` | h274 | no | buildable, but the stale read and the allocation are both inside `#if HAVE_BIGENDIAN`, and `HAVE_BIGENDIAN 0` in every config | live in source, **unreachable**: both lines compile out on little-endian |
| `415c9f7b35` | avutil/mem | no | no — gated on `HAVE_ALIGNED_MALLOC`, which is 0 | **not a defect**: it works around a blind spot in the Windows ASan runtime, not an FFmpeg memory error |
| `23aea13745` | videotoolboxenc | no | no — macOS only | **out of taxonomy**: CoreFoundation retain/release on a borrowed reference, no `av_malloc`/`av_free` and no pool API in the diff |
| `d42cd604d0` | vulkan_ffv1 | no | no — hwaccel | **out of taxonomy**: the dangling object is a Vulkan descriptor binding written by a GPU shader; 2 of the 4 touched files are GLSL |
| `4c6217477f` | nvdec | **YES — `3d5ad47c40`** | no — hwaccel | **not live.** This is the second instance of the trap: it is "absent by ancestry" yet its backport is in the tag. It would otherwise have been the only `nested`-class candidate of the six |

**`4c6217477f` also carries a near-miss worth recording.** The `fail:`/`nvdec_fdd_priv_free(cf)`
label *does* exist at the pin (`n9.0.1:libavcodec/nvdec.c:606-607`), but it belongs to a different
function where freeing `cf` is correct, because it has not yet been published into
`fdd->hwaccel_priv`. Grepping for the label alone would have produced a false LIVE — the same
one-sided-grep failure as `ops_dispatch` above.

**Only `c2173dcc32` survives, and its limitation is specific:** the port builds no HLS muxer, so the
defective path is not in our binary. A reduction of it at the allocator seam would still run — the
reductions here never execute the upstream consumer — but it must then be labelled as a reduction of
a defect **live in the pinned source and unreachable in our build**, which is weaker than fixtures 24
(FFmpeg) or 14/15 (tshark). It is also the *same measurement cell* as fixture 24: a stale struct
field read at offset 0. Noted as available, deliberately not built.

## Half (a): pool-layer candidates — all four are one change, and all need reading

`0661ef6bb3e5` (h264), `d34492955221` (hevc), `977db8c2cbfc` (av1), `0b0870dc8cfe` (vp9), all
2026-07-16, all *"Ensure that private HW data is freed early for pictures"*. Each moves one
`av_refstruct_unref(&pic->hwaccel_picture_private)` earlier in the same function.

**~~Not yet classified~~ — CLASSIFIED 2026-10-03, and the earlier suspicion was right.** The subject
says "freed early", which reads as a *resource-ordering* fix rather than a use-after-free, and the
inventory records a retraction of exactly this mistake (`a024f8c541` vp9, withdrawn because *"its own
fix classifies it as spatial, not temporal"*). Read now, against the commits' own bodies:

- **All four are REJECTED, out of class.** Each diff is a pure reordering — the same
  `av_refstruct_unref(&…->hwaccel_picture_private)` line removed and re-added a few lines earlier,
  1–2 lines net. The commit body names the dying object, and it is **not pool storage**: *"Vulkan
  video decode holds a reference to a VkSemaphore that is waited on in `ff_vk_decode_free_frame`. …
  the wait_semaphores call ends up waiting on a destroyed [semaphore]."* That is a
  destruction-ordering defect across the CPU/GPU boundary on a **driver handle**. The
  `av_refstruct_unref` is the fix's *mechanism*, not the defect's allocator — which is precisely why
  the layer must be read from the diff's semantics and not from the API that appears in it.
- **The hevc one is explicitly speculative:** its author records that another upstream developer
  suggested hevc would have a similar bug, and adds *"I couldn't reproduce a crash or VVL error in
  this area."* Hardening, not an observed defect.
- **And none is reachable:** the port's own `config_components.h` has **79** `_HWACCEL` lines with
  **0** enabled, and no `CONFIG_*VULKAN*` enabled either. (Checked two-sided: the same grep finds
  `CONFIG_H263_DECODER 1`, `CONFIG_MPEG4_DECODER 1`, so it is able to report an enabled entry.)
- All four survive the backport probe (not cherry-picked into `n9.0.1`), so liveness is not what
  rejects them — class and reachability are.

**So half (a) yields no candidate, now by reading rather than by deferral**, and FFmpeg's
pool-consumer history stands exhausted at the four the existing inventory already found.
~~**FFmpeg's nested column is empty, and that is a finding rather than a gap**: the only
`nested`-class commit in either half was `4c6217477f`, and it is backported.~~

> ### The enumeration above is WITHDRAWN — 2026-10-03. The conclusion survives, for a different reason.
>
> **What is withdrawn.** "The only `nested`-class commit in either half was `4c6217477f`" and
> "FFmpeg's pool-consumer history stands exhausted at the four" are both **false**.
> **`46f3276248`** — *"avcodec/h264_slice: clear the ER picture when starting a second field"*,
> 2026-07-30 — is `nested`-class and **live at the pin**, and this file's search missed it.
>
> **Why it was missed: filter 1.** `filter 1` is *"lifetime wording in the subject"*, and this subject
> says only "clear the ER picture". No wording filter can catch it. That is the filter's cost, stated
> here rather than left implicit — and it bounds every "exhausted" claim this file makes, including the
> ones not withdrawn.
>
> **Liveness, step 0 first (the object resolves: `git cat-file -t 46f3276248` → `commit`):**
>
>     git merge-base --is-ancestor 46f3276248 n9.0.1   -> rc=1   NOT in the pin
>     git merge-base --is-ancestor 5c66a3ab51 n9.0.1   -> rc=0   CONTROL, is in the pin
>     git log n9.0.1 --grep='cherry picked from commit 46f3276248'  -> EMPTY
>     git log n9.0.1 --grep='cherry picked from commit 4b9c4b9cfb'  -> 716d2a47c5   CONTROL fires
>
> **The storage is genuinely nested, and the pointer is INTERIOR.** `h264dec.h:570-574` declares
> `mb_type_pool`, `motion_val_pool`, `ref_index_pool` as `AVRefStructPool *`; `h264_slice.c:254` takes
> entries with `av_refstruct_pool_get`, `h264_picture.c:52-56` returns them with `av_refstruct_unref`,
> so a release reaches the pool's free list and **never the system allocator**. And
> `h264_slice.c:259` is `pic->motion_val[i] = pic->motion_val_base[i] + 4` — an interior pointer into a
> pooled entry. `ff_h264_set_erpic` (`h264_picture.c:166-187`) then copies `f`, `tf`, `motion_val[i]`,
> `ref_index[i]`, `mb_type` as **raw aliases with no refcount**. The pin's second-field `else` branch
> (`h264_slice.c:1619-1622`) does not clear that struct; `h264_frame_start` clears it for every other
> picture at `:537`. So the dangling pooled pointers are real.
>
> **Why the conclusion still holds: the dangling pointers are never DEREFERENCED at the pin.** Proved
> rather than argued, in two steps:
>
> 1. `git grep 'er\.cur_pic\|er->cur_pic' n9.0.1 -- libavcodec/` returns **three** sites: the two
>    `set_erpic` calls and one in `mpeg_er.c` for a different codec. The many other files matching
>    `cur_pic.motion_val` hold `H264Context`'s own `cur_pic`, an `H264Picture` — a different object from
>    `er.cur_pic`, which is an `ERPicture`. Only `error_resilience.c` reads the `ERPicture`.
> 2. Of its **55** `cur_pic.{motion_val,mb_type,ref_index}` accesses, 16 are in `ff_er_frame_end` and
>    all 39 others are in five `static` helpers — `guess_mv`, `guess_dc`, `h_block_filter`,
>    `v_block_filter`, `is_intra_more_likely` — whose only call sites are inside `ff_er_frame_end`
>    (checked per call site; the one other `guess_dc` hit is a log string inside `guess_dc` itself).
>
> And `h264dec.c:782`, `if (!FIELD_PICTURE(h) && h->current_slice && h->enable_er)`, wraps **both** the
> populate at `:788` and `ff_er_frame_end` at `:804`. So on a second field `ff_er_frame_end` is not
> called, and the next frame picture clears the struct at `:537`, closing the window. The only read
> reachable during the window is `er_supported()` (`error_resilience.c:823`) from `ff_er_add_slice`,
> which null-tests the stale `f` and reads the stale **scalar** `field_picture` — and `ERPicture` is
> embedded in `ERContext`, not pooled, so that read is in bounds.
>
> **Verdict: a stale-struct LOGIC defect** (`er_supported` answers for the previous picture), **not a
> memory-safety fault. Rejected as a corpus case** — there is no faulting access for any arm to catch,
> so it would measure nothing. Recorded here because "nested-class, live, and still not a case" is a
> different and more useful statement than "no nested-class commit exists", which is what this file
> said before.

## Measured, 2026-10-02: both reproduce natively, and ASan reports both

The reductions are in `ports/ffmpeg/app/src/capstone-domain/ffapp_safety.c` as fixtures 24 and 25,
whose predictions were pushed in `95aa3d340f80` before either was built. The native pair below was
run from the same reductions and is the fix-differential arm plus the `host-asan` comparator.
**Corrected 2026-10-03: "the same reductions" is false.** The native pair fills with a constant
`memset`, the fixtures with `fill`'s `v0 + i`; case 25 reads at offset 32, so the native arm
corroborated a wrong prediction. See the native bundle's own correction section.

| case | arm | plain native | with `-fsanitize=address` |
|---|---|---|---|
| 24 vvc/thread | fixed | `VERDICT FIXED field-nulled=1` | clean, rc 0 |
| 24 vvc/thread | buggy | `DEFECT-REPRODUCED same-address=1 read=5b` | **`heap-use-after-free`**, rc 1 |
| 25 ops_dispatch | fixed | `VERDICT FIXED read-from-copy=a0` | clean, rc 0 |
| 25 ops_dispatch | buggy | `DEFECT-REPRODUCED same-address=1 read=5b` | **`heap-use-after-free`**, rc 1 |

`same-address=1` is what makes this evidence rather than a crash: the storage really was freed and
reissued to a new owner, and `read=5b` is that new owner's byte where the original held `0xa0`.

**ASan firing here is the point, not a disappointment.** These are the control half; a silent ASan
would mean the reproduction is wrong. It also fires on exactly the two buggy arms and neither fixed
arm, so the comparator is shown able to say both things.

## ~~Blocked: the domain arms cannot be built on this host~~ — SUPERSEDED 2026-10-03

All three arms were built and run; the results are in
`../../ports/ffmpeg/app/results/2026-10-03-qemu-upstream-defects-heap-arms/`. The section below
was accurate for the toolchains available when it was written and is kept for that reason.

### As written, 2026-10-02

Fixtures 24 and 25 are registered for `level0`, `shrink` and `sublet`, and **none of those images
has been built**, so nothing here says what the capability machine does with these defects.

The application SDK refuses both toolchains present on this host, each for its own reason, and both
refusals are correct:

| toolchain | built | verdict |
|---|---|---|
| `llvm-capstone/llvm/cmake-build-debug` | 2026-09-24 | *"compiler emits a linear direct-call target (C-46); rebuild the toolchain"*. C-46's direct-call instance was fixed at `563e0765953e` and merged 2026-09-25, after this build |
| `llvm-capstone-tshark/llvm/cmake-build-debug` | 2026-09-25 | passes C-46, then *"application SDK requires the intcap compiler extensions; use a qualified toolchain"* |

Probed behaviourally with a positive control — a trivial file compiles for `capstone64-unknown-elf`
with both, and `__uintcap_t` compiles with neither. The qualified compiler the migration used is
named in `ports/common/application/README.md`: `compiler/sroa-keep-capability-whole` at
`7d01722aab88`, whose commit is present in this repository but had no build on this host **when
this was written**; it was built later the same day (19 minutes, 1.2 GB, Release + assertions).

**The gate must not be weakened to get a run.** `runtime/host/capstone_vm/compiler.py:63` states what
it is protecting: without the feature macro the port patches *"compile an integer-only fallback that
links successfully and faults on the first symbol/Datum access"* — an image that faults for a reason
unrelated to the fixture, which would read as a result.

**A route that does work.** The *component* port builds domain images without the application SDK:
`cmake --preset capstone-domain` on `ports/ffmpeg/buffer-pool` configures and builds `replay.dom` and
`pool-security.dom` with the 2026-09-25 toolchain. A heap corpus cut against that preset would reach
the emulator without the qualified compiler. That is the recommended next step, and it is a change to
the component's `cmake/Replay.cmake` plus a new `bug-corpora/ffmpeg/heap-repros`.

## Counts

| | |
|---|---|
| population | 1,788 commits after the pin |
| lifetime wording in the subject | 15 |
| plain-heap, read from the diff | 1 confirmed live, 1 **retracted as backported**, 3 unread |
| pool-layer, read from the diff | 4, all one change, all unclassified |
| ~~confirmed live and in class, today~~ | ~~**2**, both plain heap~~ → **1** (`vvc/thread` only) |

**Corrected 2026-10-03: one, not two.** `ops_dispatch` was backported (section 2). `vvc/thread`
survives the same probe that caught it — `git log n9.0.1 --grep='cherry picked from commit
bc46eab87c'` is empty while the identical probe for `4b9c4b9cfb` returns `716d2a47c5`, so the
negative is a tested negative and not an untested one.

The ratio is still the point, and it is now sharper: **1** usable candidate from 1,788 commits. Do
not read a small shortlist as a weak search — but do not read an unverified one as a result either.

## Upstream's own evidence that pooled memory was ASan-invisible at this pin — `e6255fb822`

Found 2026-10-03 while triaging the enumeration retraction above. **This is not a defect and not a
case.** It is an *instrument*, and it is the one piece of evidence on this question that is not ours.

    commit e6255fb822   2026-09-13
    avutil/{buffer,refstruct}: annotate pooled memory for ASan and MSan

Step 0 and liveness, same probes as everything else in this file: `git cat-file -t e6255fb822` →
`commit`; `git merge-base --is-ancestor e6255fb822 n9.0.1` → rc=1; the cherry-pick probe is empty while
the `4b9c4b9cfb` control returns `716d2a47c5`. **So it is NOT in the tree the paper compiles.**

What it does, in `libavutil/refstruct.c` — the allocator three of the four `pool-repros` cases and all
of `ports/ffmpeg/pool`'s arms sit on:

    pool_return_entry()      + if (!pool->free_entry_cb)
                             +     FF_ASAN_POISON(get_userdata(ref), pool->size);
    refstruct_pool_get_ext() + if (!pool->free_entry_cb)
                             +     FF_ASAN_UNPOISON(ret, pool->size);

and the same pattern in `buffer.c`. Upstream's message states the intent plainly: *"Poison the memory
when it enters the pool and unpoison it when it is handed out again or freed."*

**Why this matters more than our assertion of the same thing.** Every bundle in this tree that reports
ASan clean on a pooled reuse currently rests on *our* reading of why — that a return to a pool never
reaches `free`, so there is nothing for ASan to key on. `e6255fb822` is upstream adding exactly the
missing annotation, which means the absence it repairs was real at the pin, stated by the people who
maintain the allocator. That is a primary source, and it is the difference between "ASan is blind here"
as an argument and as a cited fact.

**It also makes the blindness MEASURABLE, which is the point.** A clean ASan result is only evidence
once the check is known to be able to fire (CLAUDE.md). `e6255fb822` is a positive control that can be
applied to the pin: the same reduction, built `-fsanitize=address`, against unpatched `refstruct.c`
versus a scratch copy with this commit applied. **Prediction, registered here before the run: clean on
the first, `use-after-poison` on the second.** If the second is also clean, the instrument is wrong and
no ASan-blindness number in this tree should be trusted until that is explained.

**Two limits, both from the diff rather than inferred.** The poisoning is gated on
`if (!pool->free_entry_cb)`, and upstream says why: *"RefStruct entries with an entry free callback own
allocations while they rest in the pool… LeakSanitizer does not follow pointers stored in poisoned
memory"*. So even on master today, pools with a free callback stay unpoisoned — the blindness is
**narrowed, not closed**, and a future "but ASan catches this now" objection is answered by the gate in
upstream's own patch. Second, `buffer.c` touches only `av_buffer_default_free` buffers, *"since custom
pool allocators may not be compatible"*. Both limits are upstream's words, not our framing.
