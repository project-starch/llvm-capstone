# Case 0 — `b7946098b1`, the alpha-plane overread

## Upstream

`b7946098b1`, subject **"swscale/alphablend: don't overread alpha plane on subsampled odd size"**,
one file, `libswscale/alphablend.c`, 19 insertions and 13 deletions.

The fix states the defect in a single added line:

```c
int subsample_row = y_subsample && (y << y_subsample) + 1 < lum_h;
if (x_subsample || subsample_row) {        /* was: if (x_subsample || y_subsample) */
```

so the `+ alpha_step` row — the row *below* the one being written — stops being averaged when there
is no next row. Before the fix the vertical average was taken whenever the format was vertically
subsampled, including on the last row.

## The object, and why the crossing is NESTED

`ff_sws_alphablendaway` takes the caller's planes:

```c
int ff_sws_alphablendaway(SwsInternal *c, const uint8_t *const src[], ...)
    const uint8_t *a = src[plane_count] + (srcStride[plane_count] * ysrc << y_subsample);
```

In ordinary use those are an `AVFrame`'s, and `av_frame_get_buffer` carves **every plane of a frame
from one `AVBuffer`**. Measured rather than assumed, with the probe that settled it:

```
yuva420p 16x5  buffers=1
   plane 3 data=…8e0 linesize=32 rows=5 end=…980
   BUFFER base=…080 size=3328 end=…d80
   VERDICT: one-row-past is INSIDE the buffer (slack after alpha = 1024 bytes)
```

So a read one row past the alpha plane leaves the **plane** and stays inside the **allocation** —
the same relationship a wmem chunk overread has to its block. That is what puts this case in the
inventory's nested-spatial cell, which had no upstream case before it.

The case asserts both facts itself (`CHECK` 902 and 904) rather than trusting this note.

## What is reduced, and what is not

- **The allocator is real.** The frame, its four planes and its single `AVBuffer` come from real
  `libavutil`.
- **The arms differ by the fix's own condition** and by nothing else. The frame, the plane, the row
  and the `x` index are identical in both.
- **Reduced to one average**, at `x = 0` on the last subsampled row, which is where the fix's
  condition turns false. The upstream loop runs the same average over every `x`.
- **Left out:** the horizontal half of the same overread, `a[2*x + 1]` past the row's valid width,
  which the fix also bounds with `xnext = FFMIN(2*x + 1, lum_w - 1)`. It stays inside the row's
  padded stride, so it crosses no bound any allocator or adapter owns, and including it would blur
  the row this case is for. A second case could carry it.
- **The 16-bit and byte-swapped paths** are left out; they are the same arithmetic on wider types.

## Liveness

`live_in_pin: false`. The fix is an ancestor of `n9.0.1`, and the pin carries the fixed form:
`subsample_row` occurs **5 times** in `n9.0.1:libswscale/alphablend.c` while the pre-fix
`x_subsample || y_subsample` condition occurs **0 times**, with `ff_sws_alphablendaway` present as
the positive control that the search reaches the file.

Reconstructing a pre-fix consumer shape against the shipped allocator is this tree's normal
practice — **27 of its 33 cases** are fix-reversals — and the convention is stated at
`../../memcached/allocator-repros/README.md:132-135`.

## Measured

`results/20261006-native-plane/`:

```
fixed  crossed=0 contained=1 damage=0 plane_slack=1024   VERDICT FIXED
buggy  crossed=1 contained=1 damage=1 plane_slack=1024   VERDICT DEFECT-REPRODUCED
asan   silent on BOTH arms
```

**ASan's silence is the finding.** The read leaves the plane, not the allocation, so no redzone sits
where it lands. The detector is not merely assumed to work: the same tree, the same day, records
`memcached/plain-heap-repros/00` crossing the `malloc` bound and ASan reporting it. The pair is the
project's axis, measured rather than argued.
