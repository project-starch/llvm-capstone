# Two upstream FFmpeg defect shapes, native pair and ASan (2026-10-02)

> ## RETRACTED 2026-10-03 — "live at the 9.0.1 pin" is false for case 25
>
> The original title was *"Two upstream FFmpeg defects, **live at the 9.0.1 pin**"*. **Only case 24
> is live.** Case 25's upstream fix `4b9c4b9cfb` was **backported into `n9.0.1` as `716d2a47c5`**,
> whose body reads *"(cherry picked from commit 4b9c4b9cfb56…)"*.
>
> The original liveness verdict came from grepping the pinned tree for the pre-fix line *shape* and
> finding it at `ops_dispatch.c:664`. Reading the enclosing function instead shows line 664 is on a
> path where `p` is **never freed** — it is handed to `ff_sws_graph_add_pass(…, p, op_pass_free, …)`
> and `p->pixel_bits_out` is read two lines later. The defect's real site is line 553, and it carries
> the fixed form.
>
> **What stands:** both reductions reproduce, ASan reports both buggy arms and neither fixed arm.
> Case 25 is a valid *synthetic* interior-pointer use-after-free **modelled on** `4b9c4b9cfb`, with
> `live_in_pin: false`. **What does not stand:** calling it a defect live in the version we compile.
>
> Full reasoning and the replacement liveness test (with its two controls) are in
> [`docs/ref/ffmpeg-live-defect-triage.md`](../../../../../docs/ref/ffmpeg-live-defect-triage.md).

**Question.** The triage in [`docs/ref/ffmpeg-live-defect-triage.md`](../../../../../docs/ref/ffmpeg-live-defect-triage.md)
found two lifetime defects ~~still present in the release this port compiles~~ — **corrected
2026-10-03: only case 24 is still present; case 25's fix was backported as `716d2a47c5`.** Do the
reductions in fixtures 24 and 25 actually reproduce them, and does the free-keyed comparator see them?

**Pre-registration.** Fixtures 24 and 25 and their predictions were pushed in `95aa3d340f80` before
either fixture was built or run on any arm.

## Verdict

**Both reproduce, both fixed arms hold, and ASan reports exactly the two buggy arms.**

| case | upstream | arm | plain native | `-fsanitize=address` |
|---|---|---|---|---|
| 24 | `bc46eab87c4f` vvc/thread | fixed | `VERDICT FIXED field-nulled=1` | clean, rc 0 |
| 24 | | buggy | `DEFECT-REPRODUCED same-address=1 read=5b` | **`heap-use-after-free`**, rc 1 |
| 25 | `4b9c4b9cfb56` swscale/ops_dispatch | fixed | `VERDICT FIXED read-from-copy=a0` | clean, rc 0 |
| 25 | | buggy | `DEFECT-REPRODUCED same-address=1 read=5b` | **`heap-use-after-free`**, rc 1 |

`same-address=1` is what makes these results evidence rather than a crash: the freed storage really
came back to a new owner, and `read=5b` is that new owner's byte where the first owner had written
`0xa0`. Without the premature free the read would have returned `0xa0`.

**ASan firing is the intended result.** These two are the CONTROL half of the request that produced
them: upstream defect shapes at the plain `malloc`/`free` layer — case 24 live at our pin, case 25
fixed at it — where a free-keyed tool is supposed to work. A silent ASan here would have meant the reduction was wrong. It also fires on both buggy
arms and neither fixed arm, so the comparator is shown able to say both things rather than only one.

## What these defects are

- **24**, `libavcodec/vvc/thread.c:703-715`, `ff_vvc_frame_thread_free`: the function takes
  `VVCFrameThread *ft = fc->ft` and ends with `av_freep(&ft)`. `av_freep` nulls the pointer it is
  **given**, so it clears the local alias and leaves `fc->ft` naming freed storage, which eight
  sites in that file then read. Fixed after our pin as `av_freep(&fc->ft)`, released in `n9.0.2`.
- **25**, `libswscale/ops_dispatch.c:544-664`, `compile_single`: `comp = &p->comp` is an **interior**
  pointer into `p`; `p` is freed; `comp->backend` is then read. Upstream's fix message states it:
  *"comp points into p, which is freed before comp->backend is read. Use the copy taken before the
  free."*

~~Both pre-fix lines are present in the pinned tree, at `thread.c:714` and `ops_dispatch.c:664`.~~
**RETRACTED 2026-10-03.** Only `thread.c:714` is. `ops_dispatch.c:664` is a *different site* in
`compile_single` where `p` is never freed (it is handed to `ff_sws_graph_add_pass` and
`p->pixel_bits_out` is read two lines later); the defect's site is line 553 and it carries the
**fixed** form, `c.backend->flags`. Case 25's description above is of upstream's pre-fix code, which
is **not** the code at our pin.

## CORRECTION 2026-10-03: this native pair was a FALSE CONTROL for case 25

`native-pair.c` fills the reissued object with `memset(q, 0x5B, ...)` — a **constant**. The domain
fixtures fill with `fill(p, v0, n)`, which writes **`v0 + i`**. Case 25 reads at offset 32, so the
two differ exactly there: the native arm printed `read=5b` and **agreed with a prediction that was
wrong**, while the fixture on QEMU returned `0x7b`.

Re-run on 2026-10-03 with the fixture's incrementing fill, same source otherwise:

    case 24 buggy: DEFECT-REPRODUCED same-address=1 read=5b
    case 25 buggy: DEFECT-REPRODUCED same-address=1 read=7b

So had this bundle used the same fill as the fixtures it reproduces, the prediction error would have
been caught here, a day before the QEMU run. A control that differs from the thing it controls in
the one respect that matters is not a control. The triage document's claim that the native pair was
"run from the same reductions" is withdrawn with this note.

A second, smaller overstatement in the same spirit: the ASan **fixed** arms contain no use-after-free
at all — both return before the stale read — so a clean ASan there is tautological. The comparator is
shown able to report and not report, but not on the same access.

## What this does NOT show

- **Nothing about the capability machine** — true when written, **superseded 2026-10-03** by
  `../2026-10-03-qemu-upstream-defects-heap-arms/`, where all three arms ran. As written:
  the `level0`, `shrink` and `sublet` images were never
  built, so the registered domain predictions remain predictions. The application SDK refuses both
  toolchains on this host — one emits a linear direct-call target (C-46, fixed upstream after that
  build), the other lacks the intcap extensions — and the gate is correct to refuse: without the
  feature macro the port patches compile an integer-only fallback that faults for an unrelated
  reason. The triage document records the probe, its positive control, and the component-port route
  that does build domain images with the toolchain available here.
- **Nothing about FFmpeg's own pools.** These are heap-layer defects by construction.
- **The reductions are reductions.** Real is the allocator — these are plain `malloc`/`free`, which
  is what `av_malloc`/`av_free` wrap, unreduced. Reduced is the consumer: the frame-thread object is
  64 bytes and its later reader is the probe; the compiled-op record is two 32-byte fields and the
  interior pointer is `&p->backend`. Neither drives a decoder.
- **N = 1 per cell.** The native arms are deterministic and were run once each.

Files: `native-pair.c` (the reduction both arms run), `result-lines.txt` (every line above).
