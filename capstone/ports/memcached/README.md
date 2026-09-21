# memcached 1.6.45: what allocates, counted

A census, not a port. `census-memcached.sh` reads the pinned source and
prints every number below; the source is `fetch-memcached.sh`'s, pinned by
`sources.sha256`. Run it before believing this file — if the two disagree,
the script is the authority and this file is stale.

    bash census-memcached.sh

## The allocators

Five things own storage and hand it out again without ever calling `free()`
on the path that matters. That property — reuse without free — is what the
other component ports measure, and it is what the [pymalloc](../cpython/pymalloc/README.md),
[FFmpeg](../ffmpeg/buffer-pool/README.md) and [APR](../apr/pools/README.md)
results turned on.

| allocator | lines | what it is | reuse path |
|---|---|---|---|
| **slabs** (`slabs.c`, `slabs_mover.c`, `slab_automove*.c`) | 836 + 820 + 423 | size classes over 1 MiB pages, each page cut once into fixed chunks, a LIFO free list (`slots`) per class; `mem_base` preallocation or per-page `malloc` | `do_slabs_free` pushes an item on `slots`, `slabs_alloc` pops it. A page leaves a class only through the **mover**, never through `free()`; `mem_base` pages never return at all. Chunked items — large values chained across chunks of the largest class — return chunk by chunk through `do_slabs_free_chunked` |
| **cache.c** | 140 | a generic object cache: a `STAILQ` of freed objects, `malloc` only on a miss under the limit | `do_cache_alloc` returns the first freed object **uncleared**. Three instances per worker thread: `cq` (connection-queue items), `rbuf` (`READ_BUFFER_SIZE` read buffers), `io` (`io_pending_t`) |
| **response bundles** (`memcached.c`) | ~62 | per-thread `mc_resp` objects in bundles sized to a `READ_BUFFER_SIZE` block | a slot is reused when its refcount drops; a bundle is freed only when empty and not the thread's open one |
| **bipbuffer.c** | 180 | a bipartite ring buffer | wrap-around reuse; the logger's transport and an `items.c` user |
| **extstore.c** (optional) | 993 | its own page allocator over a flash file: `free_page_buckets`, `page_version` | a page is recycled through the buckets and its **version bumped**, so a stale reference is detected by version, not by address — a revocation scheme of memcached's own, at page granularity |

## Allocator-like lifetimes, `malloc` underneath

| | what dies and comes back |
|---|---|
| **`conns[sfd]`** | a connection struct is created once per fd and reused for every later connection on that fd; `conn_free` is reached from one site, a failed init. The struct behind a closed connection is the next connection's |
| **`assoc.c`** (369) | the hash table is rebuilt by a maintenance thread; the old table is freed after migration |
| **`restart.c`** (446) | an `mmap` arena handed to `slabs_init` as `mem_base_external` and **reused across process restarts**, pointers rewritten by `restart_fixup` — a lifetime that outlives the process |

## Consumers, not allocators

`items.c` (1774) is the LRU over slabs and the place a stale item pointer would
be used; `storage.c` (1617) sits over extstore; `logger.c` (1151) over the
bipbuffer; `crawler.c` (997) walks the LRU. The **proxy** (16 files, optional,
Lua-driven) has no allocator of its own — 62 libc sites, no pool.

## The level below

163 `malloc`/`calloc`/`realloc`/`free` sites in the core (`testapp.c`
excluded), 62 in the proxy. Compare APR's 18 and pymalloc's raw fallback:
memcached is a server with many small owners, not a library with one.

## What a port of slabs would face, and how it differs from APR

`slabs.c` includes `memcached.h` whole — 1114 lines — and reaches into it for
the `item` struct (15 refs), `settings` (16), global `stats` (8), the stats
callback type and the automove stats struct. That is the shim problem, the
analogue of APR's fourteen headers.

Two things APR did not have:

- **Threads are on.** 31 `pthread_mutex` references in `slabs.c` alone. APR's
  census set `APR_HAS_THREADS=0` and the guards fell away; slabs has no such
  switch, so a freestanding build stubs the mutexes and the port states that
  it is single-threaded. The mover and the LRU maintainer are threads.
- **The mover is a lifetime event.** `slabs_mover.c` relocates or evicts every
  item on a page before the page changes class. A port that hooks only
  `slabs_alloc`/`slabs_free` would miss the moment a still-referenced item's
  storage becomes another class's page. The FFmpeg pool never moves storage;
  APR never does; pymalloc's arena release is the nearest thing and it was
  handled explicitly.

Two things worth measuring rather than porting first: `cache.c` is 140 lines
and three per-thread instances of the exact reuse-not-free shape; and
extstore's `page_version` is a version-based revocation memcached already
performs on itself, which a comparison against Sublet could read directly.

## Prior art

Outside this repository, another lane measured memcached 1.6.45's slab classes
under a workload — five rounds of set/get/delete over 20 000 keys, then an
eviction-heavy run — and recorded that no page returned: libc saw 21 frees in
a process that ended 100 000 objects, and every eviction's slot was taken
within three allocations. That is the property this census names from the
source; that measurement is the same property seen from the other side.

## Recommended order

1. **slabs**, as the httpd/APR and pymalloc ports were done: the real
   `slabs.c` through a one-source seam, hooks on `slots` push and pop, the
   mover handled as a lifetime event, a corpus case per defect shape.
2. **cache.c**, cheap and exact, as a second allocator in the same port.
3. **extstore**, optional, as a comparison rather than a port: its versioning
   is the closest thing in any of these codebases to what the discipline does.
