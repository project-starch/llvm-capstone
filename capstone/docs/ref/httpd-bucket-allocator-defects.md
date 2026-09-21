# httpd's two nested allocators, and which one to port

*Which of Apache's custom allocators is worth bringing under the discipline, and
how many consumer defects each one would make visible. Assembled 2026-09-21
against APR 1.7.4 and apr-util 1.6.3, from a full clone of `apache/httpd`.*

**Answer: the bucket allocator, by about fifteen to one.**

## Apache has two allocators, stacked

| layer | allocator | recycles without reaching `malloc`? |
|---|---|---|
| APR | `apr_pool_t` over `apr_allocator_t` | yes — `allocator_free` (`apr_pools.c:414`) pushes nodes onto a size-bucketed LIFO list, and the default `APR_ALLOCATOR_MAX_FREE_UNLIMITED` frees none of them |
| apr-util | `apr_bucket_alloc_t` | yes — `apr_bucket_free` pushes a small node onto `list->freelist`; a large one goes to `apr_allocator_free`, which is the layer above |

Three further ones are specialised and out of scope here: `apr_rmm` over shared
memory, httpd's `ap_slotmem` (plain and shm providers), and mod_http2's
`h2_bucket_beam`.

The two core ones are not peers. **The bucket allocator carves out of blocks it
obtains from APR's allocator and returns them there**, so a stale bucket pointer
sits behind *two* recycling levels, neither of which reaches `malloc`. No other
port in this repository has that shape: CPython stacks three layers but the
middle one is pymalloc, and FFmpeg's pools sit directly on `av_malloc`.

`apr_bucket_alloc` pops `list->freelist` before carving anything new, so
same-address reuse is a property of the allocator rather than of a run — the
same argument as PostgreSQL's Slab and FFmpeg's buffer pool.

## How many defects each surface would make visible

Two instruments, and the gap between them is the usual one:

| surface | diff touches allocator lifetime, message reads like a fix | says it in words |
|---|---|---|
| `apr_pool_t` consumers | **46** | 5 |
| `apr_bucket_alloc_t` consumers | **118** | 4 |

Neither number is a defect count. Both were triaged.

**Pools: 1 of 46.** The rest are leaks, MPM threading, Win32 platform work,
`mod_md` version bumps and empty `On the trunk:` subjects. The one that survived
is `9e6be73065`, mod_watchdog reusing a pool it destroyed, and it is built in
[`bug-corpora/httpd/apr-pool-repros`](../../bug-corpora/httpd/apr-pool-repros/README.md).

**Buckets: about 15 of 118.** After removing leaks (8), threading and MPM (4),
mod_http2's own strand (16) and build/doc noise (4), 86 remain, and reading
their subjects leaves roughly fifteen that carry the shape in their own words:

    106d0761c0  core: fix ap_request_core_filter()'s brigade lifetime
    d9c2352952  buckets inserted to it can be created from scpool and this pool [dies]
    38437740bb  Fix pool lifetime issues when the proxy backend connection terminates early
    c81adad105  Fix a pool lifetime issue: clean up our brigade before handing the connection back
    1c7a70c9d9  mod_proxy_http2: fixed using the wrong bucket_alloc from the backend connection
    60919177e8  Fixed a read from a deleted brigade
    4930450013  Fix bucket lifetime [issue]
    edc450c8ac  buckets associated with a subrequest having private data in the wrong pool
    6a533d0bf9  Fix winnt bucket_alloc to borrow memory from the transaction pool
    0ae93ad6f9  a separate subpool for the socket and connection members, as we destroy [...]
    bcbfbe4ace  mod_cache_socache: cached entity body corruption, buckets need setaside
    d2a1cf5f8c  a SEGFAULT by ensuring buckets buffered in the network filters get flushed
    2fe752e16b  memory issues with ranges, by keeping the original brigade untouched
    70b40483dc  a core dump in mod_cache when it stored uncopyable buckets
    826f90e639  mod_lua: fix memory handling in output filters

`106d0761c0` states the mechanism in its own message: *"For EOR it can't use a
brigade created on `r->pool`, so retain one created on `c->pool`."* Buckets move
between pools of different lifetimes — request, connection, subpool — while the
filter chain hands them across module boundaries, sets them aside, clones them
and moves them between brigades. That is where the defects are.

**And one of them names a CVE.** `d814b83206` fixes **CVE-2010-1623** in
`mod_reqtimeout`, with a non-blocking variant of `apr_brigade_split_line()`. No
other corpus in this repository has a CVE on the allocator's own surface.

## Why the pool surface is thin, and the bucket surface is not

The consumers that matter for `apr_pool_t` are third-party modules, which are in
no clone. httpd's own core is written by a small team, and a pool is tied to a
request or a connection, so "used after the pool died" rarely survives review.

Bucket consumers are `server/` and `modules/` — shipped with httpd, in the tree,
and every byte of the filter chain passes through them.

## What the port would cost

[`census-buckets.sh`](../../ports/apr/census-buckets.sh) counts it from the
source. 243 lines, 193 of them code, six public entry points. Every external
symbol it calls — five `apr_allocator_*` and four `apr_pool_*` — is defined in
`apr_pools.c`, which the pool census already builds freestanding through
[`apr_shim.h`](../../ports/apr/adapted/apr_shim.h). So the two levels port
together or not at all, and that is the point rather than an obstacle.

**The one real obstacle was the node geometry, and it is resolved.**
`SMALL_NODE_SIZE` is `APR_BUCKET_ALLOC_SIZE` plus the node header, and
`APR_BUCKET_ALLOC_SIZE` is `2*sizeof(union apr_bucket_structs)` — a union over
the bucket *type* zoo, which lives in `apr_buckets.h` and not in the file being
ported. A shim that invented that size would silently change the allocator's
block layout, its small/large split, and therefore which frees reach the
freelist at all.

So [`adapted/apr_bucket_shim.h`](../../ports/apr/adapted/apr_bucket_shim.h)
**transcribes the five structs and the union verbatim**, and
[`build-buckets-census.sh`](../../ports/apr/build-buckets-census.sh) checks that
transcription field by field against upstream's header on every run rather than
trusting it. It also answers the two questions that decide whether the port is
sound:

    1. is the transcribed geometry faithful to upstream?
       apr_bucket             identical, 7 fields
       apr_bucket_refcount    identical, 1 fields
       apr_bucket_heap        identical, 4 fields
       apr_bucket_pool        identical, 4 fields
       apr_bucket_mmap        identical, 2 fields
       apr_bucket_file        identical, 7 fields
       apr_bucket_structs     identical, 7 fields
    2. does APR_HAS_MMAP move the allocation size?
       sizeof(union) 64, APR_BUCKET_ALLOC_SIZE 128   (APR_HAS_MMAP=1)
       sizeof(union) 64, APR_BUCKET_ALLOC_SIZE 128   (APR_HAS_MMAP=0)
    3. do both levels recycle, and does anything reach malloc?
       level1_freelist_same_address=1  stale_before=0xA1 stale_after=0xB2
       level2_allocator_same_address=1
       freed_to_malloc=0
       freed_to_malloc_including_teardown=5

`APR_HAS_MMAP` does not move the size because `apr_bucket` is the union's
largest member either way — asserted by the shim's comment and **checked** by
step 2, rather than left as reasoning.

Step 3 is the structural claim, measured rather than quoted: the probe
**interposes `free()`** and counts it. Nothing reaches `malloc` while storage is
recycled, at either level, and both levels hand back the same address — the
stale read returns `0xB2`, the next owner's byte, where it wrote `0xA1`. The
last line is the honest companion: the *teardown* does free, five times.
Destroying an allocator returns memory for real. It is the **recycling** that is
invisible, and that is the distinction the corpus rests on.

## Limits

This is a census and a subject-line triage, not a reproduction. The fifteen are
candidates: at the pool surface, fourteen plausible candidates collapsed to one
once each was read at its allocation site, and that can happen again here. What
is established is the ratio between the two surfaces and the structural reason
for it, not that fifteen cases exist.
