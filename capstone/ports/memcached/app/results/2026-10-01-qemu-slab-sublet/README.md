# memcached with Sublet inside its own allocators: the slabsublet arm (2026-10-01)

**Question.** With Sublet as the runtime heap, memcached's own allocators stayed unprotected: a slab
item was bounded to its 1 MiB page and never revoked, so the safety fixtures 9 and 10 returned on
every heap arm (`../2026-10-01-qemu-safety/`). Does the server still serve correctly once the
component port's lifetime hooks run inside it, and do those two fixtures then fault?

**Pre-registration.** Patch 0006, the glue, the build script, the runner changes, the predictions
(`host/safety-expect.txt`, the plan's S1 section) were pushed to lane branch `slab-sublet` in
1cb89fb5ddf7 before the first boot. Nothing in the predictions has changed since.

## Verdict

**Every run is as pre-registered: oracle 6 of 6, safety 60 of 60 and then 18 of 18 with fixture 11,
the report and the guard as predicted.**

| | mode 0 (spatial) | mode 1 (sublet) |
|---|---|---|
| oracle, 3 runs in one boot | transcript identical to native (1,931,207 bytes), identity 128, exit 0, stderr empty | the same |
| fixture 9, write into the neighbouring slab item | **bounds fault** at the neighbour's first data byte, 3/3 boots | **bounds fault**, 3/3 |
| fixture 10, read a removed and reused item through the stale pointer | returns `a0015b` (same address, the new item's byte), 3/3 | **temporal fault**: the stale pointer reloads untagged at the printed target, 3/3 |
| fixtures 1–8 | the sublet arm's verdicts, 3/3 | the same, 3/3 |
| fixture 11, a chunked item (700 KB: header plus one 512 KiB chunk) removed and reallocated, read through the old chunk pointer | returns `b0015b` (same chunk, the new item's byte), 3/3 | **temporal fault** at the printed target, 3/3 |

On every other arm fixture 9 returns `9000ee` and fixture 10 returns `a0015b`: those are the negative
controls, already recorded, and the two flips above are the positive controls that the hooks are
live.

**The report run** (`MC_SLAB_SUBLET_REPORT=1`, mode 1, oracle still identical):
`pages=9 chunk_releases=20 chunk_reuses=20 object_releases=127 object_reuses=111 backing_used=9699776 metadata=1305600`.
Nine slab pages were carved, twenty chunks went through release and reissue, and 127 cache objects
were released and 111 reissued, each a revoke in mode 1. The twenty fit the script: phase 1's
explicit frees (failed add/replace/cas, replace, append and prepend, the incr wrap, expired gets,
delete, the meta delete, `flush_all`) and two flushed keys reclaimed in the background; the ~330
stored values are never freed (`curr_items 328`, `evictions 0`).

**Hooks exercised by these runs:** page backing and carve, chunk-at, chunk issue (chunked items
included), the non-chunked chunk release, object backing, issue and release. **Not exercised:** the
chunked-item release (the oracle never frees its 600 KB and 900 KB values: they are still linked at
`stats`), page discard (`memory_release` returns early without `slab_reassign`), object discard (no
cache limit, no `cache_destroy`), and the mover (not started). Fixture 11 was registered after that
audit (pushed in 6c73bc73fb61, before its first boot) for the chunked release: it frees a chunked
item (a header in class 3 plus one attached 512 KiB chunk in class 37) through `item_remove` →
`do_slabs_free_chunked`. The chunk returns to its class's free list and is reissued at the same
address to the next item of the same shape (`same-address=1`; its alias is bounded to the chunk,
`[cc180000,cc200000)`). That can only happen if the chained-chunk release ran, and the header's
release runs unconditionally before it. It ran on a rebuilt safety image, 286e2211…, with fixtures
9 and 10 alongside, 3 boots per mode: 18 of 18 as predicted, the fixture's output byte-identical
across boots.

A direct count, one more boot in mode 0 with the report line on: fixture 10's server reports
`chunk_releases=1 chunk_reuses=1`, fixture 11's `chunk_releases=2 chunk_reuses=2`. The two releases
are the header's and the chunk's, counted by the ledger itself, with no address argument needed.

What fixture 11 does not show: the header's own reuse (its new address is not printed); which of
the two renews, the release's or the reissue's, killed the old pointer in mode 1 (both revoke, as
in fixture 10); and the loop's second iteration, since one chunk was attached where a real 700 KB
value carries a smaller tail chunk as well. The item is never linked or filled: this is the path
from `item_free` down, not a server-side delete. The pre-registered fixture source needed a
one-line build fix before it compiled (an `#include "items.h"` that `memcached.h` already supplies,
and `items.h` has no guard); the fix is committed with these results. Page discard, object discard
and the mover remain unexercised.

**The guard fires.** One run of the oracle image with no `MC_SLAB_SUBLET_MODE` in the environment:
the server printed `MCAPP-SLAB-SUBLET fail 903` and exited 97 before listening, and the harness
recorded no transcript. So a run that loses the variable cannot pass as either mode.

**Item bounds, read from the fixtures' own output.** On the plain sublet arm an item's data pointer
reads `bounds=[c8800000,c8900000)`, its page; here it reads `[cc0fff00,cc0fffd0)`, its 208-byte
class-3 chunk.

## What the arm is

- **Patch 0006** carries the allocators component port's patch 0002 onto the application's
  `slabs.c` and `cache.c`, the same hooks at the same transitions: page backing and discard, page carve, chunk-at, chunk issue
  and release (plain and chunked), object backing, issue, release and discard, and the metadata
  heap for the slab lists and cache control blocks. Each hook is behind `MC_CAPSTONE_SLAB_SUBLET`
  with upstream's line in the `#else`. `main` calls the adapter's init before `slabs_init`.
- **The adapter** is the component's `leases.c`, `metadata.c` and `authority.c`, compiled unchanged.
  `leases.c` takes the item layout from the app's `memcached.h` through a one-line
  `mc_slabs_shim.h`.
- **The glue** (`src/slab-sublet/mcapp-slab-sublet.c`): the 64 MiB payload lent LINEAR by the Sublet
  runtime heap at `HEAP_LOG 27` (a 128 MiB pool); 16 MiB of metadata from `malloc`; one mutex around
  every hook; the mode from `MC_SLAB_SUBLET_MODE`, required (an absent or other value ends the
  server with code 903); a report line only on request. In mode 1 a chunk's first filing by the
  page split (carved, never issued) is not revoked; every release and issue after that is.
- **Gates** in `host/build-slab-sublet.sh`: DRIFT (patch 0006's 20 hook calls over 14 hooks equal
  patch 0002's by name and count; `--drift-selftest` shows a patch missing one hook is refused.
  It does not check placement: an audit moved a release ahead of the reads it must follow and the
  gate passed, so placement rests on reading the patch), HEADERS (from the
  compiler's dependency lists: `authority.c` read `runtime/include/sublet/sublet.h`, `leases.c` the
  app's `memcached.h`), ARM (adapter present in both images, absent from every other arm's, images
  differ from the plain sublet arm's), LINK.

## Run settings, and what they cost

- `-m 48 -o no_slab_reassign`, native and domain alike. The adapter's page half is 48 MiB, and the
  page mover is not hooked: it walks a page by pointer arithmetic, which per-chunk bounds refuse.
- `CAPSTONE_REV_NODES=16777216`: carving is eager, about two revocation nodes per chunk, which a
  silicon pool of 65,536 cannot hold. **This arm is a QEMU result.**
- The domain's stop after SIGTERM takes 4.4 s against 2.3 s on the plain sublet arm.
- The launcher was 63e8a39a… (PR #176's fix). The runtime was the one `deps/env.sh` built for this
  tree, which is newer than the one the M5 images linked. The temporal faults are QEMU's untagging
  of a revoked capability on reload (ISSUES Q-11); the QEMU binary is the same by path in every
  boot and is not hashed per run.
- Every oracle verdict line in `result-lines.txt` reads `[shrink]`: the script labelled by its
  default arm name. The image hashes, not the label, say what ran; the label is fixed for later runs.

## Two sentences in the pre-registration were wrong

1. The plan's S1 section says "the server itself never calls `cache.c`". It does: every worker's
   read buffers and IO objects come from `cache.c` through `do_cache_alloc`/`do_cache_free`
   (upstream `memcached.c:398-432, 1047-1199`, and `cache_alloc`/`cache_free` at `thread.c:314,327`;
   the caches are made at `thread.c:454,471`). The report run's object counters show it. The
   sentence came from a `grep … | head` that cut its list at ten lines, before `thread.c`. The
   consequence is in the arm's favour: each 16 KiB read buffer is bounded and, in mode 1, revoked on
   every return to its cache.
2. The W2 prediction says "the chunked release hook runs on every arm". It did not run in any
   oracle run: the script never frees its large values. Fixture 11 was added for that path.

The predictions did not rest on either sentence and are unchanged.

Files: `result-lines.txt` (every run's lines), `SHA256SUMS` (the two images, verifiable from
`$MC_WORK`).
