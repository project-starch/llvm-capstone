# Case 3 — `5c66a3ab51`, VVC: a non-reference output frame releases its pooled side tables

## The upstream defect

    commit 5c66a3ab51  2024-09-15
    avcodec/vvc: Fix output and unref a frame which isn't decoding yet

Upstream's own account of the mechanism, from the commit message:

> It's possible for ff_vvc_output_frame to select current frame to output. If current frame is
> nonref frame, it will be released by ff_vvc_unref_frame.

The whole fix is one line in `ff_vvc_set_new_ref`:

    -        ref->flags = VVC_FRAME_FLAG_OUTPUT;
    +        ref->flags = VVC_FRAME_FLAG_OUTPUT | VVC_FRAME_FLAG_SHORT_REF;

## Why that one line is load-bearing, and only for a non-reference picture

At the pin, `ff_vvc_set_new_ref` reads (`n9.0.1:libavcodec/vvc/refs.c:251-257`):

    if (s->no_output_before_recovery_flag && (IS_RASL(s) || !GDR_IS_RECOVERED(s)))
        ref->flags = VVC_FRAME_FLAG_SHORT_REF;
    else if (ph->r->ph_pic_output_flag)
        ref->flags = VVC_FRAME_FLAG_OUTPUT | VVC_FRAME_FLAG_SHORT_REF;   <- the fix

    if (!ph->r->ph_non_ref_pic_flag)
        ref->flags |= VVC_FRAME_FLAG_SHORT_REF;

The second `if` puts `SHORT_REF` back **only when the picture is a reference picture**. So for a
picture with `ph_pic_output_flag` set *and* `ph_non_ref_pic_flag` set, the pre-fix flags are
`VVC_FRAME_FLAG_OUTPUT` alone. That is the combination the commit message names, and it is the
condition `case.c` writes out.

`ff_vvc_unref_frame` (`n9.0.1:libavcodec/vvc/refs.c:44-73`) then does:

    frame->flags &= ~flags;
    if (!(frame->flags & ~VVC_FRAME_FLAG_CORRUPT))
        frame->flags = 0;
    if (!frame->flags) {
        av_frame_unref(frame->frame);
        ...
        av_refstruct_unref(&frame->tab_dmvr_mvf);
        ...
        av_refstruct_unref(&frame->rpl_tab);
        ...
    }

so outputting such a frame clears its only flag, the `!frame->flags` branch runs, and both side
tables go back to their pools while the decoder is still using the frame.

The tables are taken from `AVRefStructPool`s at `n9.0.1:libavcodec/vvc/refs.c:153` and `:157`:

    frame->tab_dmvr_mvf = av_refstruct_pool_get(fc->tab_dmvr_mvf_pool);
    frame->rpl_tab      = av_refstruct_pool_get(fc->rpl_tab_pool);

Flag values are `n9.0.1:libavcodec/vvc/refs.h:28-32` — `OUTPUT (1<<0)`, `SHORT_REF (1<<1)`,
`CORRUPT (1<<4)`.

## Liveness at the pin

**Not live, and the reason is ancestry rather than a backport.**

    $ git merge-base --is-ancestor 5c66a3ab51 n9.0.1 ; echo $?
    0                     # in the pin
    $ git merge-base --is-ancestor bc46eab87c n9.0.1 ; echo $?
    1                     # NOT in the pin -- the control, a real commit either way

The control matters: a first attempt used a sha from another repository and got `rc=128`, which is
"bad object" and not "not an ancestor", so it proved nothing. Both shas above are real FFmpeg
commits, and the probe returns 0 for one and 1 for the other, so it discriminates.

The fixed form is present verbatim at `n9.0.1:libavcodec/vvc/refs.c:254`. So the case
re-introduces the reverse of the fix, which is the tier all three other cases in this corpus are in
(`expect_live_in_pin: {"false": 4}` after this one).

Worth stating because a sibling investigation got caught by the other failure mode: a fix can be
*absent by ancestry* and *present by content* as a cherry-pick, which is how
`swscale/ops_dispatch`'s `4b9c4b9cfb` reached `n9.0.1` as `716d2a47c5` and forced a retraction. Here
ancestry settles it directly, so the cherry-pick probe is not the deciding instrument — noted so
nobody reads its absence as an omission.

## Why this one was worth adding

`AVRefStructPool` is one of the two FFmpeg pool allocators the paper ports and it carried **zero**
cases; all three existing cases are on `AVBufferPool`. It is a genuine recycling pool, not a wrapper:
`av_refstruct_unref` pushes the entry onto `pool->available_entries` (`refstruct.c:230-231`) and the
next `av_refstruct_pool_get` pops the same one back (`:258-261`), with the header stating objects
"will be reused for subsequent `av_refstruct_pool_get()` calls". So a premature release here never
reaches the system allocator, which is exactly the boundary this corpus exists to test.

## What the reduction does and does not keep

**Kept, unreduced:** `libavutil/refstruct.c` and `libavutil/buffer.c` are upstream's, compiled by the
port (`ports/ffmpeg/buffer-pool/cmake/Replay.cmake:21-22` lists both), so the recycling the case
depends on is the allocator's own behaviour and not the driver's.

**Reduced:** `ff_vvc_unref_frame`'s flag arithmetic is transcribed rather than called; the frame
struct carries only `plane`, `tab_dmvr_mvf`, `rpl_tab` and `flags`; no decoder is built, and the
triggering flag combination is written out instead of being reached through a bitstream. The same
model-consumer/real-allocator split as cases 0-2.

**Not claimed:** nothing about hardware-accelerated decode (`hwaccel_picture_private` is also
released by that branch, and is out of scope here), and nothing about the frame-threading race that
`ccd391d6a3` is about.

## Files

`case.c` — the reduction, both arms in one source selected by `fixed`.
`case.json` — the claims, with every arm this case cannot carry declared `"not written"` rather than
omitted.
