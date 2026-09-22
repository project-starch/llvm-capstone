# Provenance

**Upstream fix:** `60919177e8`, 2002-01-27. **Consumer:**
`server/protocol.c, ap_rgetline()`. **CVE:** `NO VERIFIED CVE` — the commit carries none and no
advisory database was searched for this entry.

## The defect, in upstream's words

> In the "folding" case, there was an ap_get_brigade() call after the brigade had been destroyed. I noticed this while debugging a memory leak that showed up while testing the httpd with ab.

## What ends the lifetime

The brigade's destroy, returning its pool storage.

The storage is **the brigade holder, whose storage the destroy gave back**.

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
