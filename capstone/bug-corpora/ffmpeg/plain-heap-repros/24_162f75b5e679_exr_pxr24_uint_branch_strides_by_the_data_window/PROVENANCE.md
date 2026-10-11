# 162f75b5e679 — avcodec/exr: use tile dimensions in pxr24 UINT case

## The defect

`td->tmp` is sized from `td->channel_line_size * td->ysize`, where `td->channel_line_size = td->xsize * s->current_channel_offset` — i.e. by the **tile** width. The `EXR_UINT` branch of `pxr24_uncompress` lays out its four byte planes and advances the cursor using `s->xdelta`, the full data-window width, while the sibling FLOAT and HALF branches correctly use `td->xsize`. For any tile narrower than the data window the cursor walks too far per row and reads past the decompressed payload.

## Upstream defect

- **Fix:** `162f75b5e679`, *"avcodec/exr: use tile dimensions in pxr24 UINT case"*, `libavcodec/exr.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin uses the tile width. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
                ptr[1] = ptr[0] + s->xdelta;
                ptr[2] = ptr[1] + s->xdelta;
                ptr[3] = ptr[2] + s->xdelta;
                in     = ptr[3] + s->xdelta;

                for (j = 0; j < s->xdelta; ++j) {
```

## The fix

```c
                ptr[1] = ptr[0] + td->xsize;
                ptr[2] = ptr[1] + td->xsize;
                ptr[3] = ptr[2] + td->xsize;
                in     = ptr[3] + td->xsize;

                for (j = 0; j < td->xsize; ++j) {
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no EXR file and no pxr24 decompression. The buffer is sized by the tile width as upstream's is, the two widths are upstream's, and the first byte outside is probed.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
