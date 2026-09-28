# Track B, vidstab: a pool defect in FFmpeg's real code, inside a third-party library — predictions

Registered and pushed **before either image was first built**. Plan:
`docs/plans/2026-09-25-ffmpeg-full-port-and-sublet.md`, Track B. The af_join run is
`../2026-09-29-trackb-afjoin/`.

## What runs

Until now, the vidstab defect has run only as the buffer-pool port's probe case 38: a reduction that
models libvidstab as the one field that matters, a retained pointer. Here, libvidstab itself runs.

- **libvidstab is compiled into the image.** Release 1.1.1, pinned by URL and SHA-256 in
  `deps/libvidstab.json`. `build-domain.sh` compiles every source its CMake build compiles, with the
  image's own flags, and OpenMP, SSE2 and ORC off. FFmpeg's configure finds it through a pkg-config
  file, as it would on any system.
- **The graph is real.** `buffer` → `vidstabtransform` → `buffersink`, 16x16 `yuv420p`. That is
  libavfilter as configured with `--enable-gpl --enable-libvidstab --enable-filter=vidstabtransform`.
  The transforms file, eight frames of zero motions, is written by the fixture into the guest's
  `/tmp` and read by the filter's own `config_input`.
- **The frames are pool frames.** Every frame comes from FFmpeg's `yuv4` decoder, so every plane is
  an `AVBufferPool` buffer handed out by `avcodec_default_get_buffer2`.
- **The defect is the exact reverse of its fix.** Fixture 22 links `vf_vidstabtransform.o` as 9.0.1
  ships it. Fixture 23 links the same file with upstream's fix `316531e61c` reversed, ahead of
  `libavfilter.a`. The reversal is applied from the fix's own diff (`trackb/vidstab-316531e61c.diff`)
  at fuzz 0. Three gates stop the build:
  - `make`'s own command, rerun, must reproduce the archive's object byte for byte;
  - the out-of-tree route the revert is compiled by must also reproduce it from the shipped text;
  - re-applying the fix to the reverted file must give back the shipped file byte for byte.

## The sequence, and why it reaches the defect

1. **Frame 1** (luma 0xA0) goes in while the fixture keeps its own reference, so the graph's frame
   is **not writable**.
   - As shipped, the pad's `AVFILTERPAD_FLAG_NEEDS_WRITABLE` copies it first, and libvidstab
     copies the source into storage of its own.
   - Reverted, the filter takes the separate-buffer path, and `vsTransformPrepare` keeps a
     **shallow pointer** to frame 1's plane (`td->src = *src`).
2. The fixture drops its reference. **Frame 1's buffer goes back to the decoder's pool**; on
   poolsublet, that return is a revoke.
3. **The victim** (luma 0x77), decoded next and held by the fixture, takes that buffer again.
4. **Frame 2** (luma 0x5B) goes in as the graph's only reference, so it **is writable**, and takes
   the in-place path. Reverted, `vsTransformPrepare` finds `td->src` already set, skips its own
   allocation, and `vsFrameCopy` copies frame 2 into the stale pointer, which is the victim's plane.

The stale write is libvidstab's own. So the fixture's touch line is printed immediately before
frame 2 is handed to the graph, and its target is frame 1's plane, read while that plane was live.
Afterwards the fixture reads the victim's first luma byte **through the victim's own frame**.

**Native control, already run** (x86-64, ASan, stock pools, the same sequence and the same revert):

- shipped: victim 0x77, same address;
- reverted: victim **0x5b**, same address.

ASan reports nothing in either case, because a pool never frees the plane.

## Predictions, N = 3 per cell

The mark is `(same-address << 8) | victim's first luma byte`.

| | poolstock (FFmpeg's pools as shipped) | poolsublet (the Sublet port of the pools) |
|---|---|---|
| **22, fix present** | RETURN `1600177` | RETURN `1600177` |
| **23, fix reverted** | RETURN `170015b`: the victim, a live frame, overwritten with frame 2's 0x5b | **FAULT temporal** after the touch line, at frame 1's plane |

**The poolsublet fault site.** It is predicted at libvidstab's copy, not at a fixture access:
the image's `memcpy`, storing through the stale capability (`stc a5, 0(a3)`, reloaded untagged).
The value in the fault line is expected to be frame 1's plane, which is the printed target. The
classifier accepts a temporal fault only after the touch line and only at the target.

**Same address.** It is predicted on both arms. The port keeps the pool's LIFO order: af_join's
fixture 19 printed `same-address=1` on poolsublet.

## What would refute it

- 23 completing on poolsublet.
- 22 faulting on either arm.
- 23 on poolstock not showing the overwrite, whether by `same-address=0` or by a byte other than
  0x5b. The fixture would then not have created its condition, and the poolsublet fault could not
  be attributed to the defect.
- 23 on poolsublet faulting before the touch line, or at an address other than frame 1's plane.

## Build

`FFAPP_HEAP=sublet FFAPP_POOL=sublet|stock FFAPP_EXTRA_CONFIGURE="--enable-avfilter --enable-gpl
--enable-libvidstab --enable-filter=vidstabtransform --enable-decoder=yuv4"`, in a work directory of
its own. The compiler (`3979abd8`) and emulator are those of the af_join run, and app patch 0004
stays applied. The C-50, budget and layout gates apply unchanged.
