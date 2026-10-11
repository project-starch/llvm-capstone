# Provenance

**Upstream fix:** `461fb22053`, *"avfilter/af_join: fix wrong loop bound in
buffer dedup (use-after-free)"*, 2026-05-25, backported to `release/8.1` as
`64fd93e361`. **Not live at the 9.0.1 pin** — the fix's removed line is present
verbatim in `n8.0`, `n8.1` and `n8.1.1` and gone from `n8.1.2` onwards, checked
by searching it in each tag's own copy of the file. What the pin governs is the
fidelity tier, not whether the fixture builds: what the port pins is the
allocator, and the defective loop is transcribed here.

**Consumer:** `libavfilter/af_join.c`, `try_push_frame()`. **CVE:**
`NO VERIFIED CVE` — the commit carries none and no advisory database was
searched for this entry. Reported by two external security researchers, named
in the upstream trailer and deliberately not repeated here.

## The defect

`try_push_frame()` copies each output channel's data pointer out of an input
frame and tracks the buffer behind it, so the output can take a reference to
every distinct buffer it now points into. The whole fix is the bound of that
test:

```c
-        if (j == i)
+        if (j == nb_buffers)
```

While every channel brings a new buffer the two are equal. The moment one
channel shares a buffer with an earlier one, `nb_buffers` stops advancing with
`i`, and from then on a genuinely new buffer ends the search at
`j == nb_buffers != i` and is never tracked. No reference is taken for it. When
the input frames are released, that storage returns to the pool while the
output frame's `extended_data` still names it.

Three output channels over two inputs is the minimum that shows it:

| i | buffer | search ends at | test `j == i` | tracked |
|---|---|---|---|---|
| 0 | input 0 | `j = 0` | 0 == 0 | yes |
| 1 | input 0, already seen | `j = 0` | 0 != 1 | no — correct, a duplicate |
| 2 | input 1, **new** | `j = 1` | 1 != 2 | **no — the defect** |

## Why the storage is pool storage

`av_frame_get_plane_buffer()` returns the `AVBufferRef` behind a plane. For
frames a decoder produced that reference came from `av_buffer_pool_get()` via
`avcodec_default_get_buffer2` (`libavcodec/get_buffer.c:196`), so the last unref
runs `pool_release_buffer` and the payload goes onto the pool's LIFO freelist at
the same address rather than to `free`. The fixture records that as
`reuse_same_address=1`.

## Real and reduced

Real: `libavutil/buffer.c`, the pool itself, compiled unmodified through the
port. Transcribed: the dedup loop, including the defective comparison. Reduced:
the filter graph, two audio inputs and `ff_filter_frame`, none of which changes
what the allocator is asked to do, and the part of `AVFrame` the loop touches —
the port extracts the allocator, not `libavutil/frame.c`.
