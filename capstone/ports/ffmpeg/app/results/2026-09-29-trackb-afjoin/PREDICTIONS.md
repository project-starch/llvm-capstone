# Track B, af_join: a pool defect in FFmpeg's real code — predictions

Registered and pushed **before either image was first built**. Plan:
`docs/plans/2026-09-25-ffmpeg-full-port-and-sublet.md`, Track B.

## What runs

Until now, the pool defects have run as the buffer-pool port's probe cases 36–38: reductions that
transcribe the defective loop around a pool. Here the defect runs in FFmpeg's own code.

- **The graph is real.** Two `abuffer` sources feed `join` (`inputs=2:channel_layout=3.0:
  map=0.0-FL|0.0-FR|1.0-FC`), which feeds `abuffersink`. That is libavfilter as configured with
  `--enable-avfilter --enable-filter=join`.
- **The frames are pool frames.** Each input is decoded by its own `pcm_s16le_planar` decoder, so
  every plane is an `AVBufferPool` buffer handed out by `avcodec_default_get_buffer2`, not a
  buffer the fixture allocated.
- **The defect is the exact reverse of its fix.** Fixture 18 links `af_join.o` as 9.0.1 ships it.
  Fixture 19 links the same file with upstream's fix `461fb22053` reverted, one token
  (`j == nb_buffers` → `j == i`), ahead of `libavfilter.a`. Two gates stop the build:
  - `make`'s own command, rerun, must reproduce the archive's `af_join.o` byte for byte;
  - the out-of-tree route the revert is compiled by must also reproduce it from the shipped text.
  So the two images differ by that token and nothing else.
- **The mapping triggers the defect.** Output channel 1 repeats input 0's buffer and channel 2
  brings input 1's. With the revert, the output frame takes no reference to input 1's buffer. The
  join frees its input frames, the buffer goes back to its decoder's pool, and decoding input 1's
  next packet reissues it. The touch reads output channel 2.

## Predictions, N = 3 per cell

| | poolstock (FFmpeg's pools as shipped) | poolsublet (the Sublet port of the pools) |
|---|---|---|
| **18, fix present** | RETURN `120005b`: input 1's first packet (0x5b), a different buffer for the next packet | RETURN `120005b` |
| **19, fix reverted** | RETURN `1300177`: the next packet's byte (0x77), in the same buffer | **FAULT temporal** at the touch: the buffer was revoked when it went back to its pool |

The bottom-right cell is the protection. The top-right cell is what shows the port is not faulting
on something else: the same graph, with the fix present, completes.

## What would refute it

- 19 completing on poolsublet.
- 18 faulting on either arm.
- 19 on poolstock not showing the reuse: `same-address=0`, or a value other than 0x77. That would
  mean the fixture did not create its condition, and the poolsublet fault would then not be
  attributable to the defect.

## Build

`FFAPP_HEAP=sublet FFAPP_POOL=sublet|stock FFAPP_EXTRA_CONFIGURE="--enable-avfilter
--enable-filter=join --enable-decoder=pcm_s16le_planar"`, in a work directory of its own, on the
same compiler (`3979abd8`) and emulator as the pool port's v2 run. The C-50 and budget gates
apply unchanged.
