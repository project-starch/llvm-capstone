# A full FFmpeg port and a full Sublet port — plan and Phase 0 measurements (2026-09-25)

Branch `ffmpeg-full`. This supersedes nothing: `2026-09-23-ffmpeg-full-app-port.md` remains the
record of M0–M5 and M6, and this plan starts where it stops.

Two tracks, because two different things are incomplete.

- **Track A — port FFmpeg's own pools onto Sublet.** Today's pool arms are not a port of FFmpeg's
  allocator, they are a substitute for it: payloads are diverted into the port's own
  `pool-allocator.c` (2048 fixed exact-size blocks, bump-carved, never coalesced) in a separate
  host region; `pool->alloc` is never called; custom allocators are refused (`ff2_fail 125/126`);
  there is no per-pool senior handle, so `uninit` frees entries one at a time; and the grant is
  `REV_TRANSFERRED` where SQLite's is `REV_BORROWED` under a monitor-held handle. The SQLite port
  is the standard to meet (`ports/sqlite/sublet/README.md`): the allocator keeps its own policy,
  its blocks come linear from the level below, and one revoke at the level above destroys
  everything below it.
- **Track B — run the three pool defects as FFmpeg's real code**, not as the hand-written
  reductions of probe cases 36–38.

## The correction that shapes Track B

**All three defects are already FIXED at the pinned 9.0.1.** Read in the pristine tree, not taken
from a doc:

| defect | 9.0.1 as shipped | re-introduction |
|---|---|---|
| af_join | `af_join.c:472` `if (j == nb_buffers)` | one token → `i` |
| h264_refs | `h264_refs.c:159,180` `memset(…, sizeof(H264Ref) * (32 - len))` | 2 hunks → the `len < ref_count` form |
| vidstab | `vf_vidstabtransform.c:258-266` — the removed path survives only as a comment | restore ~26 lines |

So each defect is re-introduced by the exact reverse of its upstream fix, in a test-only arm, and
the clean 9.0.1 build is the matched control.

## Phase 0 — MEASURED, 2026-09-25

Compiler `3979abd8e9a3`, FFmpeg 9.0.1, `FFAPP_WORK=/tmp/capstone/ffmpeg-sizeprobe{,2}`.
Configure adds `--enable-avfilter --enable-filter=join --enable-filter=aformat
--enable-decoder=pcm_s16le --enable-decoder=h264 --enable-parser=h264 --enable-demuxer=h264`.

| probe | arm | libavfilter | `code_len` | block needed |
|---|---|---|---|---|
| 1 | level0 | not built | 7,164,624 (6.83 MiB) | 8 MiB |
| 2 | sublet | built | 6,266,832 (5.98 MiB) | 8 MiB |

Committed baseline for comparison (2026-09-24 revalidation, M5): level0 3,624,240 · shrink
3,623,536 · sublet 2,725,488 · pool0/2 2,963,392.

Archive attribution, from the probe's own build:

| | bytes | note |
|---|---|---|
| `libavcodec.a` | 5,352,768 | 106 members |
| — `h264*` members | **3,229,648** | 23 members: the expensive part |
| — `mpeg*` members | 264,912 | 14 members |
| `libavutil.a` | 2,060,498 | |
| `libavformat.a` | 1,037,854 | |
| `libavfilter.a` | **461,736** | `af_join`, `af_aformat`, and the core, which includes `buffersrc.o`, `buffersink.o` and `framepool.o` |

**Conclusions that bind the rest of the plan.**

1. **h264 alone breaks the old 4 MiB ceiling; libavfilter is cheap.** af_join is affordable at any
   time; h264 is what forces the block size up.
2. **The sublet arm is the one to size from**, not level0: its heap is a granted region rather than
   `.bss`, so sublet *with* libavfilter and h264 is smaller than level0 without them.
3. **`buffersrc`/`buffersink` are core**, always built with libavfilter, so the Phase 2 driver needs
   no extra `--enable-filter` for them.

### Two defects found and fixed while measuring

- **`build-domain.sh` built a hardcoded library list** (`libavutil libavcodec libavformat`), so
  `--enable-avfilter` *configured* libavfilter and never built or linked it. The image would have
  contained no filter at all while the build reported success. The set now follows configure's own
  `config.mak`, and a missing archive is an error rather than a silent omission.
- **The runner passed no `cma=`.** Since buildroot `2b8ad05` (pinned by #97) a domain block beyond
  4 MiB is served from CMA, but this port's kernel is built `CONFIG_CMA_SIZE_MBYTES=0`, so nothing
  is reserved unless the kernel is told to. A large image would simply have failed to load — which
  reads as a stall, not as a refusal. `run-qemu.sh` now sizes and passes it by the tshark port's
  formula, with the same ≤1792M refusal.
- `ORDER_CEILING` is now `FFAPP_ORDER_CEILING_MB` (default 4, unchanged). It stays a gate: raising
  it is only valid together with `cma=` **and** a CMA-capable guest module. The shared buildroot
  checkout is still at `d04bd83`, *before* that change, so large images need a private rootfs.

### The h264 finding that gates Phase 3

The upstream fix `1886c3269d` names its testcase **`poc10.bin`**, reported with a proof-of-concept
bitstream by a browser vendor and never published. The stale read is the **error-concealment** path
(`h264dec.c:84-92`), reached when slices disagree on `ref_count`; its own FIXME says exactly that.
`ref_list` is `[2][48]`; the fix clears `[len, 32)`, the defect clears only `[len, ref_count)`.

Both readers checked first — `fill_colmap` and `ff_h264_direct_ref_list_init` (`h264_direct.c`) —
are bounded by `ref_count` and are **not** the defect's reader.

**Therefore h264 needs a malformed stream, not a clean encode.** Phase 3 is gated on producing one
within a written budget; if none reaches the reader, case 37 stays the reduction and the gap is
recorded. There is no local H.264 encoder (no ffmpeg, x264, openh264 or gstreamer on this host), so
the generator is a further external dependency, host-only and never linked into the domain.

## Track A design

The recipe is the one `capstone/sublet/sublet.h` already documents:

| operation | primitive |
|---|---|
| block from the level below | `sublet_take_linear` — mrev, the block stays LINEAR |
| carve an object | `sublet_split`, then `sublet_take` — mrev, delin |
| free an object | `sublet_give` |
| destroy the pool | `sublet_give_to` on the handle senior to the children: **one revoke**, whatever hangs below dies |

**The missing piece, now written.** `sublet_heap.c` exposed only `malloc/free/calloc/realloc`, all
returning delinearised aliases, and a nested allocator cannot carve an alias. Added, as the
equivalent of SQLite's `sqlite3MallocLinear`:

- `__capstone_sublet_malloc_linear(n, &slot)` — carves exactly as `malloc` does (the carve is now
  factored into `sh_carve_block`, shared by both, so the arms keep one buddy policy and one set of
  counters) and hands the block out with `sublet_take_linear`: the region stays LINEAR in the
  caller's slot and `sh_cap[i]` keeps the senior handle.
- `__capstone_sublet_free_linear(base)` — one revoke on that handle reclaims the lent region **and
  everything the borrower carved from it**, then the buddy merge runs unchanged. Keyed by base
  because the lender holds no alias, and no scrub, because `sublet_give`'s write-through is the
  scrub exactly when a linear child makes the revoke return UNINIT.

Both hooks then apply to FFmpeg's two pools, which have the same shape — a freelist, a get, a
return, and a teardown that today frees entries one at a time:

- `libavutil/buffer.c`: `av_buffer_pool_get` / `pool_release_buffer` / `buffer_pool_flush` +
  `av_buffer_pool_uninit`;
- `libavutil/refstruct.c`: `refstruct_pool_get_ext` / `pool_return_entry` / `pool_free_entry` +
  `pool_free`.

FFmpeg's pools grow one entry at a time rather than taking one fixed block, so the pool's senior
handle must be taken **before its first entry**, and every entry carved below it — otherwise the
one-revoke teardown does not hold. That handle lifecycle is the first thing to build.

## Verification, per defect

Native fix-differential first (the corpus runner already does it), then in the domain:

| | mode 0 / bounds only | Sublet |
|---|---|---|
| **defect re-introduced** | completes or miscomputes | **faults at the labelled stale access** |
| **9.0.1 as shipped** | completes | **completes** |

The bottom-right cell is the one that matters most: it is what shows Sublet is not faulting on
something else. Arms differ by exactly the reverted hunks.

Per phase: predictions pre-registered and pushed to `ffmpeg-full` before the first boot; N = 3 per
cell; a results folder with result lines, `SHA256SUMS` and the compiler/emulator hashes; a
claim-auditor before any conclusion is written. Every new fixture gets a positive control the day
it is written, and the revocation-node budget (65,532 per boot, never reclaimed on QEMU) is
predicted per arm before the first boot — an exhausted pool mid-decode reads exactly like a fault.

Standing caveats on every result: QEMU only (**Q-11** — the deployed silicon lets a stale access
retire); M6 and the one-translation-unit `gp`-captable question are untouched; **I-12** guest
stalls are live.
