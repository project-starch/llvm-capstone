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

`spatial`, `sublet`, both PoisonCap arms and `cheribsd-revocation` are declared **predictions**.
`spatial` and `sublet` are predicted to **complete**: the crossing is inside one `av_malloc`, so a
per-allocation bound is in bounds for it, and nothing in the port narrows frame planes today. **A
discriminating reading needs a plane-narrowing adapter**, which is port work rather than a case —
the same relationship `chunks` has to the wmem corpus. Until it exists this row measures the gap
rather than closing it, which is what the inventory says.

## Running it

```sh
bash runners/run-native.sh [OUT_DIR]
```

Exit 0 means the plain pair reproduced **and** the sanitiser stayed silent. Exit 75 is an
infrastructure failure and is never a verdict. The build links real `libavutil`; override
`FFPLANE_SRC` and `FFPLANE_BUILD` to point at another one.
