# e9e6fb879835 — avcodec/tdsc: Check tile_size

## The defect

The raw-tile path copies `h` rows of `w * 3` bytes out of `ctx->tilebuffer`, which was allocated for `tile_size` bytes read from the stream. `w` and `h` come from a different field, and nothing related `3 * w * h` to `tile_size`, so an under-sized tile is read past its end.

## Upstream defect

- **Fix:** `e9e6fb879835`, *"avcodec/tdsc: Check tile_size"*, `libavcodec/tdsc.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin carries the check. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
        } else if (tile_mode == MKTAG(' ','W','A','R')) {
            /* Just copy the buffer to output */
            av_image_copy_plane(ctx->refframe->data[0] + x * 3 +
                                ctx->refframe->linesize[0] * y,
                                ctx->refframe->linesize[0], ctx->tilebuffer,
                                w * 3, w * 3, h);
```

## The fix

```c
            if (3LL * w * h > tile_size)
                return AVERROR_INVALIDDATA;
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no TDSC stream and no reference frame. The buffer is allocated for the stream's tile_size and the copy's length is the geometry's, with the read reduced to the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
