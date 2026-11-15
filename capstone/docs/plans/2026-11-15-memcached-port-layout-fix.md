# memcached 1.6.45: the slab lifetime patch with upstream's layout, pre-registration 2026-11-15

Committed before any measured run. A reading that differs from what is written here is a finding;
this file is not edited afterwards.

## Problem

Run-5 (`docs/plans/2026-11-15-memcached-established-workload.md`, records in
`ports/memcached/app/results/2026-11-15-established-reuse/`) measured the lifetime patch
`ports/memcached/app/patches/memcached-1.6.45-0006-slabs-and-cache-sublet-lifetimes.patch` under
mc-crusher. At the slab boundary the protected arm differs from stock by 0.0056 to 0.0084 of the
reuses (stock noise floor 0.0016 to 0.0030). The cause is the patch's layout:

- `item` and `item_chunk` gain a 16-byte `sublet_parent` at offset 0. Every chunk grows by 16 bytes
  (class 3: 208 to 240, class 19: 8,448 to 9,088), each page holds 6 to 9 % fewer chunks, and the
  full classes evict about 33,500 items instead of about 31,100.
- `sublet_parent` holds the page's generation, a reference whose bounds cover every chunk of the page.
  It sits inside the bounds of every chunk the patch hands out. A live item pointer can load it,
  reach every other chunk of the page, and end them all with one CREVOKE.

Two more references with authority beyond the object sit in memory the client receives:

- `do_slabs_alloc` pops a chunk off the free list and leaves its link (`next`) in place. The link
  holds the allocator's pointer to the next free chunk, whose bounds cover the page (or the generation).
- `do_cache_alloc` pops an object and leaves its `STAILQ` link in place. The link holds the next free
  object's renewed reference, which stays live when that object is handed out to another owner.

## Change

1. `item` and `item_chunk` keep upstream's layout: patch 0006 no longer touches `memcached.h`.
2. Each chunk is a CDERIVE child of its page, the reference that the class keeps in `slab_list` as
   `memory_allocate` returned it. `do_slabs_alloc` derives the chunk from the free-list entry, which is
   the allocator's own pointer into the page, and clears the entry's link before the chunk leaves.
3. `slabs_free` ends the caller's chunk with CREVOKE, then finds the page that holds the chunk's
   address by address range over `slabclass[id].slab_list` (then over every class, because a chunked
   item's header comes from a class other than the one its owner passes). `do_slabs_free` continues
   through that page reference moved to the chunk, and writes its bookkeeping into the released chunk
   as upstream does. `do_slabs_free_chunked` takes back each chained chunk the same way. The lookup
   reads only `slab_list`: no allocation, no new structure, no read through the caller's reference.
4. `slabs_finalize_page_move` renews the page's lifetime with CRENEW when the page leaves a class. This
   ends every chunk the page handed out, and the page goes on under the renewed reference. If CRENEW
   has no lifetime left, the page ends (CREVOKE) and leaves the slab allocator.
5. With `-L`, `memory_allocate` derives each page it cuts from the preallocated block as a child of the
   block, so that renewing one page ends no other page's chunks.
6. `do_cache_alloc` clears the free-list link of the object it hands out. The rest of `cache.c`
   (CRENEW at `cache_free`) stays as it is.
7. The mover's three whole-chunk copies return to upstream's `memcpy`. The `assoc_find` re-lookup of a
   linked item stays.
8. `experiments/study/reuse-hooks/memcached/memcached-1.6.45-reuse-gap-hooks.patch` is rebased onto the
   new context: the same observation points. Its old hunks adapted the call sites that patch 0006
   changed. The stock sources of `slabs.c`, `cache.c`, `slabs_mover.c` and `memcached.h` with the hooks
   applied stay identical to run-5's (checked with `unifdef -UMC_CAPSTONE_SUBLET -DEXP_REUSE_GAP`).

## Checks

### Static layout

A probe compiled in both arm trees of `build-reuse.sh`, with each arm's configure flags, records
`sizeof(item)`, `offsetof(item, data)`, `sizeof(item_chunk)` and `offsetof(item_chunk, data)`.
Prediction: equal in both arms. The slab class table follows from `sizeof(item)` and the settings,
so equal sizes give equal chunk sizes and chunks per page.

### Unit test with a negative control

`ports/memcached/app/tests/sublet-layout/`: one program linked against `slabs.c` and `cache.c` through
the allocators port's Sublet build (`MCP_SUBLET=ON`, the corpus build), once with patch 0006 as of
5060011844e (old) and once with this change (new), run on the trap-route platform.

| mode | what it does | old | new |
|---|---|---|---|
| `scan` | allocate a slab chunk and a cache object, load every 16-byte slot inside each one's bounds, report every tagged reference whose bounds reach beyond the object | LEAK at chunk offsets 0 (generation) and 16 (free-list link) and at cache offset 0 (next object); exit 1 | no LEAK; exit 0 |
| `stale` | free a chunk with `slabs_free`, then read through the caller's reference | fault cause 25 | fault cause 25 |
| `bounds` | read one byte past a small chunk | fault cause 28 | fault cause 28 |
| `move` | fill one page of a large class, move it to the global pool with `slabs_finalize_page_move`, use a chunk of the class's other page and a chunk of another class that takes the moved page, then read a chunk of the moved page | other chunks work; last read faults cause 25 | the same |
| `move-prealloc` | `move` with pages cut from one preallocated block (`slabs_init` with a block passed in) | not run | the same as `move` |

### Measurement

Exactly the run-5 workload and settings: `-l 127.0.0.1 -p 21299 -U 0 -m 32 -t 4 -A -o slab_automove=0`,
`EVENT_NOEPOLL=1`, the oracle, mc-crusher 7ca3f14c with `mc-crusher-established.conf` (240,000
commands), the run-5 load client binaries, `run-established.sh`, kit `eval/kit-run-4`, QEMU
9289114e06, kernel Image-0133b9e21cd, firmware f-keep. Images from `build-reuse.sh` on this branch.
The protected arm runs two repetitions. The run-5 stock repetitions count as this comparison's stock
only if the new stock image is byte-identical to run-5's (sha256 f272d96abda0...). Otherwise stock
runs two repetitions as well. Records go to `~/asr-work/eval/run-6/reuse/memcached-<arm>-<rep>/` with
`provenance.txt`. Each run's wall time is recorded.

### Corpus

The only bug corpus built with patch 0006 is `bug-corpora/memcached/allocator-repros`, arm
`virtual-nested-pools` (`shared/build-cases.sh capstone-application` with `MCP_SUBLET=ON`), with its
control 90. No trap-route run of it exists (run-2 to run-4 ran memcached's plain-heap and
plain-temporal repros, which do not build with patch 0006). The arm runs on the trap-route kit twice,
with the old and with the new patch, to `~/asr-work/eval/run-6/corpora/memcached-allocator-repros-<arm>/`.
The app's safety fixtures (`host/build-sublet.sh`, `memcached-safety.dom`) also build with patch 0006,
but their runner (`host/run-virtual.py`) needs the module-based route, which the trap route replaced.
They are not rerun.

## Predictions

1. **Output.** Every run's oracle transcript has sha256 ee1522c8e0ec..., every load completes with
   processed = 240,000, the observer reports error 0 at both boundaries, and `capstone-job` records exit 0.
2. **Static layout.** The four sizes are equal in both arms.
3. **Dynamic layout.** In every protected run, every slab class's `chunk_size`, `chunks_per_page` and
   `total_pages` and `total_malloced` equal run-5 stock (classes 3, 8, 13 and 19: chunks of 208, 704,
   2,192 and 8,448 bytes, 5,041, 1,489, 478 and 124 per page, 1, 2, 4 and 17 pages). Slab issues are
   60,352, distinct starts 12,368 and reuses 47,984, as in run-5 stock. Evictions, in total and in each
   of classes 3, 8, 13 and 19, lie within the range of the stock runs widened by that range's width on
   each side (run-5 stock alone: total 31,047 to 31,216, so 30,878 to 31,385).
4. **Slab distribution.** The median bucket is 1 in every run. Every stock x protected reusediff is at
   most the largest within-arm reusediff of this program at the slab boundary in run-5 and run-6 (stock
   and protected pairs together; run-5 alone gives 0.0030).
5. **Cache distribution.** The median bucket is 1 in every run. Every stock x protected reusediff is at
   most the largest within-arm reusediff at the cache boundary in run-5 and run-6 (run-5 alone: 0.0074).
6. **Unit test.** As in the table.
7. **Corpus.** New patch: cases 00 to 04 CAUGHT with cause 25 at `read_probe`, 06 and 07 CAUGHT with
   cause 28, 08 CAUGHT with cause 28 at its declared site, control 90 faults. Case 05 changes from
   NO-READING to CAUGHT with cause 28 at `write_probe`: its setup writes a sentinel into byte 0 of the
   neighbouring live item, which held the parent capability under the old patch and made the fixed
   run's `slabs_free` fault. Old patch on the same platform: as on 2026-10-11, eight CAUGHT and case 05
   NO-READING.
