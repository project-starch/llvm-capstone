# httpd's two nested allocators, and which one to port

*Which of Apache's custom allocators is worth bringing under the discipline, and
how many consumer defects each one would make visible. Assembled 2026-09-21
against APR 1.7.4 and apr-util 1.6.3, from a full clone of `apache/httpd`.*

**Answer: the bucket allocator, by eight to one.**

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

**Buckets: 8 of 118, with 3 more unresolved.** After removing leaks (8),
threading and MPM (4), mod_http2's own strand (16) and build/doc noise (4), 86
remained; reading their subjects left fifteen, and reading the fifteen messages
in full left eight.

The two purest are defects in the **allocator identity itself**, which no other
corpus here has:

| fix | what it is |
|---|---|
| `1c7a70c9d9` | *"using the wrong `bucket_alloc` from the backend connection when sending data on the frontend one. This caused crashes or infinite loops"* |
| `d2a1cf5f8c` | buckets *"created with the bucket allocator of the backend connection"*, which *"either gets destroyed ... or it will be used again by another frontend connection that wants to recycle the"* connection |

`apr_bucket_free` reads `node->alloc` and pushes onto **that** list, so a bucket
freed through the wrong allocator lands on a freelist whose owner does not own
the block — and the next `apr_bucket_alloc` on that list hands it out.

Six more are lifetime defects on pool or bucket storage, each naming its own
mechanism:

    106d0761c0  "For EOR it can't use a brigade created on r->pool"
    d9c2352952  "buckets inserted to it can be created from scpool and this
                 pool can be freed before this brigade"
    c81adad105  clean up the brigade before handing the backend connection
                back to the connection pool
    60919177e8  "a read from a deleted brigade ... an ap_get_brigade() call
                 after the brigade had been destroyed"
    4930450013  "Fix bucket lifetimes so that they don't live longer than
                 their brigades"
    edc450c8ac  "buckets associated with a subrequest having private data in
                 the wrong (i.e., subrequest) pool, leading to a segfault
                 later in processing the main request"

Three are unresolved and need their diffs read: `38437740bb` and `bcbfbe4ace`
(both turn on whether a missing setaside leaves the storage pool-backed), and
`70b40483dc` (uncopyable pipe buckets).

Five were rejected, and one rejection corrects an earlier claim in this file:

| fix | why not |
|---|---|
| `6a533d0bf9` | pchild **exhaustion**, not a lifetime defect |
| `0ae93ad6f9` | its message says memory **leak** |
| `826f90e639` | brigade iteration in **constant memory** |
| `2fe752e16b` | CVE-2011-3192, the byte-range **memory exhaustion** DoS |
| `d814b83206` | **CVE-2010-1623 is a denial of service**, not a temporal defect. An earlier version of this file offered it as "a CVE on the allocator's own surface". That was wrong: the fix's own changelog entry reads *"Fix a denial of service attack against mod_reqtimeout"*, and its new helper is commented *"to avoid DoS by high memory usage"*. **This corpus has no CVE.** |

## Why the pool surface is thin, and the bucket surface is not

The consumers that matter for `apr_pool_t` are third-party modules, which are in
no clone. httpd's own core is written by a small team, and a pool is tied to a
request or a connection, so "used after the pool died" rarely survives review.

Bucket consumers are `server/` and `modules/` — shipped with httpd, in the tree,
and every byte of the filter chain passes through them.

## The port, done 2026-09-22

The bucket allocator now runs in a Capstone domain with a Sublet adapter,
carried by the pool port (`ports/apr/pools`, `-DAPRP_BUCKETS=ON`): the pool
port lends each 8 KiB block linear, the bucket allocator's pieces are split
from it and die with it, and the freelist a freed node used to hold moved to
the adapter's records. Two patches, six hunks, every decision upstream's.
The eight cases measured through it: `spatial` completes all eight; `sublet`
faults at the labelled read on all eight -- case 4, the corpus's one
reuse-not-free case, once its reduced consumer declares the connection
handback as the lender's epoch, the operation upstream expresses only as a
copying discipline; stock CheriBSD with libc revocation on completes all
eight through the same port build. Records and mechanism per case:
[`bug-corpora/httpd/bucket-repros/`](../../bug-corpora/httpd/bucket-repros/README.md).
The paragraph below is the estimate that preceded it, kept as written.

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
