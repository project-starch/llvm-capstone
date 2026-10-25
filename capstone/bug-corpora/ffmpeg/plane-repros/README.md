# FFmpeg plane-crossing spatial defects — the nested row

Upstream FFmpeg defects whose access **leaves a frame plane while staying inside the frame's single
allocation**. The third FFmpeg corpus, and the one the inventory's nested-spatial cell needed:

| corpus | boundary | axis |
|---|---|---|
| [`../pool-repros`](../pool-repros/README.md) | storage an `AVBufferPool` handed out and took back | temporal, nested |
| [`../subobject-repros`](../subobject-repros/README.md) | two **members** of one `av_malloc` | spatial, **not** nested |
| **this one** | two **planes** of one `AVBuffer` | spatial, **nested** |

**That the planes come from one buffer is measured, not assumed.** A `YUVA420P` frame from
`av_frame_get_buffer` reports `buf[1] == NULL` — a single `AVBuffer` — with **1024 bytes of slack
after the alpha plane inside it**. So a read that leaves the plane stays inside the allocation,
exactly as a wmem chunk overread stays inside its block, and per-`malloc` bounds are in bounds for
it. The case asserts both facts rather than relying on this note.

## Shapes

| shape | cases |
|---|---|
| a vertical average reads one row past a plane, inside the frame's single AVBuffer | 0 |

## The case

| case | upstream | the crossing |
|---|---|---|
| **0** | `b7946098b1` `libswscale/alphablend.c` | on the last subsampled row, `a[2x + alpha_step]` reads the alpha row **below the plane's last** |

## Measured, 2026-10-06

[`results/20261006-native-plane/`](results/20261006-native-plane/result-lines.txt):

| arm | buggy | fixed |
|---|---|---|
| `native-fix-differential` | `crossed=1 contained=1 damage=1` → DEFECT-REPRODUCED | `crossed=0 contained=1 damage=0` → FIXED |
| `native-detect` (ASan) | **silent** | **silent** |

The arms differ by exactly the fix's own condition,
`subsample_row = y_subsample && (y << y_subsample) + 1 < lum_h`, which is false on the last row.

**ASan's silence is the finding, not a gap.** The read leaves the *plane*, not the *allocation*, so
no redzone sits where it lands. [`../../memcached/plain-heap-repros`](../../memcached/plain-heap-repros/README.md)
records the opposite reading for the opposite reason — there the crossing leaves the `malloc` bound
and ASan reports it. The two side by side are the project's axis, measured rather than argued:

> a crossing that leaves the allocation is seen by every tool; one that stays inside it is seen by
> none, and only an adapter that knows the inner layer can separate them.

## Arms not measured here

`spatial`, `sublet` and `cheribsd-revocation` are declared **predictions**.
`spatial` and `sublet` are predicted to **complete**: the crossing is inside one `av_malloc`, so a
per-allocation bound is in bounds for it.

> **RETRACTED 2026-10-06.** What stood here said *"a discriminating reading needs a plane-narrowing
> adapter, which is port work rather than a case"*, implying this row is a gap waiting on an
> adapter. **It is not, and the measurement says so.** `av_frame_get_buffer` pads to
> `FFALIGN(height, 32)` before filling the plane pointers (`libavutil/frame.c`), so for the
> `YUVA420P` 16×5 frame this case uses:
>
> | | bytes |
> |---|---:|
> | the alpha plane as the consumer sees it, `linesize × height` | **160** |
> | the alpha plane's own allocated extent, `av_image_fill_plane_sizes` at `FFALIGN(5,32)=32` | **1024** |
> | the offset this case reads at | **160** |
>
> So the read is **inside upstream's own plane extent**, not merely inside the buffer. An adapter
> narrowing each plane to FFmpeg's own layout — the faithful thing to narrow to, and what
> `wm_narrow` does for wmem — **would not catch this crossing**. Only a bound tighter than
> upstream's own allocation would, and that bound faults legitimate code: SIMD tail reads and
> `av_image_copy_to_buffer` both work over `padded_height` on purpose.
>
> **This row is therefore a measured non-gap.** It is the same shape as the sub-object corpus's
> "nothing we have catches this", one level out: the crossing leaves a bound the *consumer* keeps
> in its head and no *allocator* ever sets. Nothing is owed here; what was owed was this
> measurement.

## Running it

```sh
bash runners/run-native.sh [OUT_DIR]
```

Exit 0 means the plain pair reproduced **and** the sanitiser stayed silent. Exit 75 is an
infrastructure failure and is never a verdict. The build links real `libavutil`; override
`FFPLANE_SRC` and `FFPLANE_BUILD` to point at another one.
