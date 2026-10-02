# Two upstream FFmpeg defects, live at the 9.0.1 pin: native pair and ASan (2026-10-02)

**Question.** The triage in [`docs/ref/ffmpeg-live-defect-triage.md`](../../../../docs/ref/ffmpeg-live-defect-triage.md)
found two lifetime defects still present in the release this port compiles. Do the reductions in
fixtures 24 and 25 actually reproduce them, and does the free-keyed comparator see them?

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
them: real upstream defects at the plain `malloc`/`free` layer, where a free-keyed tool is supposed
to work. A silent ASan here would have meant the reduction was wrong. It also fires on both buggy
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

Both pre-fix lines are present in the pinned tree, at `thread.c:714` and `ops_dispatch.c:664`.

## What this does NOT show

- **Nothing about the capability machine.** The `level0`, `shrink` and `sublet` images were never
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
