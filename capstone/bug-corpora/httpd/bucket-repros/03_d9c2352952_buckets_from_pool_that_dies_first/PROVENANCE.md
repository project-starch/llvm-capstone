# Provenance

**Upstream fix:** `d9c2352952`, 2013-12-11. **Consumer:**
`modules/ssl, the brigade outliving scpool`. **CVE:** `NO VERIFIED CVE` — the commit carries none and no
advisory database was searched for this entry.

## The defect, in upstream's words

> Cleanup the bb brigade, because buckets inserted to it can be created from scpool and this pool can be freed before this brigade. POSSIBLE (but as yet unconfirmed) fix for crashes seen with threaded servers, e.g. PR 50335.

## What ends the lifetime

Apr_pool_destroy of scpool while the brigade still names the payload.

The storage is **bucket payload allocated from scpool**.

## Why it is invisible to a malloc-level tool

`apr_bucket_free` pushes a small node onto the bucket allocator's own LIFO
freelist and stops there; a large one goes to `apr_allocator_free`, which is
APR's size-bucketed list, and the default `APR_ALLOCATOR_MAX_FREE_UNLIMITED`
frees none of those either. **Two recycling levels, and neither reaches
`malloc`.** The fixture measures that rather than quoting it: `free()` is
interposed and counted, and the case prints `freed_to_malloc`.

## Real and reduced

Real: `apr-util/buckets/apr_buckets_alloc.c` and `apr/memory/unix/apr_pools.c`,
both upstream byte for byte through the port's shims, with the bucket node
geometry transcribed verbatim and checked against upstream's header on every
build. Reduced: brigades, filters, connections and requests — to the holder and
the storage, which is what every one of these defects turns on.
