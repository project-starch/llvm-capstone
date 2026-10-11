# Provenance

**Upstream fix:** `d2a1cf5f8c`, 2007-12-15. **Consumer:**
`modules/proxy/mod_proxy_http.c and the network filters`. **CVE:** `NO VERIFIED CVE` — the commit carries none and no
advisory database was searched for this entry.

## The defect, in upstream's words

> these buckets have been created with the bucket allocator of the backend connection. This allocator either gets destroyed if conn->close is set or the worker address is not reusable ... or it will be used again by another frontend connection that wants to recycle the connection.

## What ends the lifetime

Apr_bucket_alloc_destroy, or the allocator being handed to the next frontend connection.

The storage is **nodes from the backend connection's allocator, still buffered downstream**.

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
