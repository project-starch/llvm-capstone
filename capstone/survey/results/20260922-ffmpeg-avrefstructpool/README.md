# The FFmpeg pool nobody measured is the busier one

`ffmpeg-avrefstructpool` on 9.0.1, which is already the revision
`capstone/ports/ffmpeg/buffer-pool` pins. The buffer pool was recorded in
the same run and is kept beside it, because the size difference is the
finding.

## The two pools, decode rung, repetition 1

| | AVBufferPool | AVRefStructPool |
|---|---:|---:|
| allocations | 27,000 | **108,001** |
| reuses | 26,979 | **107,957** |
| before backing release | 100.0 % | 100.0 % |
| distinct addresses | 21 | 44 |
| most objects at one address | 1,291 | 4,514 |

libavutil carries two pooled allocators and the survey has only ever
recorded the first. On this workload the second hands out four times as
many entries. A reader of the existing FFmpeg row would take 27,000 for
the program's pooled traffic and be wrong by a factor of five once both
are counted.

Neither lets anything reach malloc before the storage serves again, and
both concentrate on a few dozen addresses. The shapes are alike here,
unlike memcached's pair, and that is worth saying as plainly as the
difference: a per-allocator survey is not an argument that every level
looks different, it is what lets either answer be stated.

## Repetitions

| rep | refstructpool allocations | reuses | max at one address | bufpool allocations |
|---|---:|---:|---:|---:|
| 1 | 108,001 | 107,957 | 4,514 | 27,000 |
| 2 | 108,001 | 107,957 | 4,513 | 27,000 |
| 3 | 108,001 | 107,957 | 4,511 | 27,000 |

The counts are identical and only the busiest address moves, by three
objects, which is the decoder's thread scheduling and not the workload.

## Where the instrument sits

`hook-refstruct.py` in the manuscript's `experiments/a1/ffmpeg/` places it
in `libavutil/refstruct.c` the way `hook.py` places the other in
`buffer.c`: anchored on text, refusing a missing anchor, a doubled anchor
or a second run. Two sites, both read at 9.0.1.

The release is the branch of `pool_return_entry` that pushes the entry
back onto `available_entries`. The address counted is `get_userdata(ref)`,
the pointer the consumer was handed, not the `RefCount` header in front of
it, so both sides of a lifetime name the same bytes. The hand-out is the
point in `refstruct_pool_get_ext` where the free-list branch and the
fresh-allocation branch converge, so one site counts each exactly once.

No bulk release. `refstruct_pool_uninit` walks `available_entries` and
frees each through `pool_free_entry`, and every entry on that list is
already dead, so nothing live dies at teardown. The level runs with
`A1_NO_BULK`, the same reasoning as memcached's object cache. The other
branch of `pool_return_entry`, taken once the pool is uninited, also goes
to `pool_free_entry` and is a level-0 release, which the interposed malloc
family records itself.

## Provenance

| | |
|---|---|
| FFmpeg | 9.0.1, pinned in `experiments/a1/ffmpeg/sources.sha256` |
| workload | decode rung, `ffmpeg -i in.mkv -f framemd5 -` |
| wiring | `ffhook-refstruct.inc`, level name `refstructpool`, `A1_NO_BULK` |
| companion pass | `experiments/results/W2/20260922T021602Z-ffmpeg`, three arms, three repetitions |
| oracle | the shipped arm's framemd5 output, matched byte for byte |
| compiler | gcc 13.3.0 |
| raw | `raw/`, hashed in `raw/SHA256SUMS` |

## Limits

One rung. The arm targets in the Makefile are absolute paths, so asking
for `work/inst-refstruct/bin/ffmpeg` finds no rule and plain `make` builds
all three.
