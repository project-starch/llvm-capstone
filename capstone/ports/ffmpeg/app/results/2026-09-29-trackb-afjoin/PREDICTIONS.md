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

## Addendum, 2026-09-29: the first run refuted its own control, and why

The first boots (images built from 4a9e6ed46132) never reached the defect. Fixture 18, the shipped
code, faulted *before its touch* on both arms, inside `avfilter_graph_config`. The registration
above names that outcome as a refutation ("18 faulting on either arm"), so the run said nothing
about af_join.

**Cause: a compiler defect, not FFmpeg and not the pools.** It was bisected in the domain:

- A configure-only diagnostic (fixture 30: `abuffer` → `abuffersink`, no join) faults the same way.
  So does the same diagnostic on the **level0** heap, which never revokes. The untagged pointer was
  therefore never freed: its tag was stripped.
- The same graph runs clean natively under AddressSanitizer, with the predicted output (0x5b).
- An instrumented `avfiltergraph.c` shows `link->incfg.channel_layouts` with tag 1 when read
  directly, and tag 0 when read through `FF_FIELD_AT(void *, m->offset, link->incfg)`
  (libavutil/internal.h:79).
- The compiler (`3979abd8`) gives `*(T *)((char *)&obj + off)` alignment 1, and legalizes the
  capability load into 16 byte loads, integer stores to a stack temporary, and an `ldc` from it,
  which drops the tag. A ten-line reduction reproduces it at -O1 and -O3, and has been reported to
  the compiler lane.

**Workaround, and its matched pair.** App patch 0004 gives `FF_FIELD_AT` the pointee's alignment
(`__builtin_assume_aligned`), under `__CAPSTONE__` only. On the reduction the load becomes one
`ldc`. On the level0 heap, fixture 30 faults without the patch and returns `1e00001` with it:
same arm, same image source, one patch apart.

**Predictions unchanged.** Fixtures 18 and 19 are re-run with 0004 applied, N = 3, against the
table above. 0004 changes every FFmpeg build of the app port. The committed results predate it, and
M5's decode path is re-checked on the rebuilt images in the same batch.

## Addendum 2, 2026-09-29: the rerun with 0004, and a fixture correction before the next one

With 0004 applied, M1-M5 MATCH on both arms, with the flip control firing. Of the 12 fixture cells,
9 came out as registered: fixture 18 returns `120005b` on both arms, and fixture 19 on poolstock
returns `1300177` (same address, 0x77), each N = 3.

The other 3, fixture 19 on poolsublet, faulted *before the touch*. The first read of the stale
pointer was the fixture's own diagnostic `show()` of the output's third channel, which reads the
capability's bounds (`lcc`). capstone-qemu faults on `lcc` of an UNTAGGED operand. The logged operand,
`0x102c07e00`, is exactly input 1's plane, which the fixture printed while it was live. So the plane
had been revoked by the time the output frame reached the consumer, which is the defect caught
earlier than the registered site. That is still a DIFFERS against the registration, and it is
recorded as one.

The fixture is corrected so that nothing of the output's plane is read before the touch: the target
address is input 1's plane, read while live, as every other fixture here takes its target before
the free. Predictions unchanged. Fixture 19 on poolsublet is expected to FAULT temporal at the touch;
18 and poolstock's 19 are expected unchanged. Re-run N = 3 on both arms.

## Addendum 3, 2026-09-29: the claim audit's corrections, and one more run before any result is written

A claim audit of the addendum-2 run upheld the 2x2 and the compiler diagnosis, and found four
things this file says that are wrong or too strong:

1. **"Images built from 4a9e6ed46132" (addendum 1) is wrong.** The build at that commit failed its
   own gate (`AF_JOIN GATE: make printed no compile command`). The first boots' images were built
   from working-tree fixes to `build-domain.sh`, committed afterwards in `150887238a68`.
2. **"The two images differ by that token and nothing else" (above) holds for the objects, not
   the images.** The reverted object's disassembly differs from the shipped one in the dedup loop
   alone, and the shipped object matches the archive member. But fixture 19 linked its object
   ahead of the libraries while fixture 18 took the archive's copy at its natural position, so
   every later symbol moved: 10,062 symbol lines differed between the images.
3. **M5's re-check "on the rebuilt images" (addendum 1) cannot test 0004.** The M5 image contains
   no libavfilter code. It shows that the rebuilt libraries decode unchanged, and nothing about
   `FF_FIELD_AT`.
4. **The compiler defect's locus.** The `align 1` is upstream clang's (#152575, fork commit
   `5569bf26f009`). The same compiler gives `align 1` for x86-64 too. The Capstone defect is the
   backend lowering an under-aligned capability access to integer bytes, which drops the tag. It
   does this for stores as well: `stc` to a temporary, then 16 `sb`.

**Before any result is written, 18 and 19 run once more, N = 3 per arm, with two changes:**

- **Fixture 18 links af_join's SHIPPED object ahead of the libraries,** exactly as 19 links the
  reverted one. The two images then differ in af_join's bytes and the fixture id alone.
- **Both print the output frame's reference count** (`out-refs`: non-NULL `out->buf[]`), so the
  missing reference is observed rather than inferred from the pair. Only pointers are compared
  with NULL; nothing they point to is read.

Predictions: the marks are unchanged (18: `120005b` on both arms; 19: `1300177` on poolstock,
FAULT temporal at the touch on poolsublet). `out-refs` is predicted to be **2** for 18 and **1** for
19 on both arms, because the reverted dedup loop never adds input 1's buffer.
