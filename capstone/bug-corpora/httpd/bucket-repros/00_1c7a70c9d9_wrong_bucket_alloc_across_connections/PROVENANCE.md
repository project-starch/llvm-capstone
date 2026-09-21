# Provenance

**Upstream fix:** `1c7a70c9d9`, 2023-06-01. **Consumer:**
`modules/http2/mod_proxy_http2.c`. **CVE:** `NO VERIFIED CVE` — the commit carries none and no
advisory database was searched for this entry.

## The defect, in upstream's words

> fixed using the wrong "bucket_alloc" from the backend connection when sending data on the frontend one. This caused crashes or infinite loops in rare situations.

## What ends the lifetime

Apr_bucket_alloc_destroy on the backend connection, which returns its blocks to apr's allocator.

The storage is **a node from the backend connection's apr_bucket_alloc_t, held by the frontend**.

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
