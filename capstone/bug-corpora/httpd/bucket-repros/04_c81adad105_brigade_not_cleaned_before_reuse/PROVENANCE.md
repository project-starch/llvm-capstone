# Provenance

**Upstream fix:** `c81adad105`, 2010-11-04. **Consumer:**
`modules/proxy/mod_proxy_http.c`. **CVE:** `NO VERIFIED CVE` — the commit carries none and no
advisory database was searched for this entry.

## The defect, in upstream's words

> Fix a pool lifetime issue: Make sure we clean up our brigade before we hand the backend connection back to the connection pool.

## What ends the lifetime

The connection, and with it the allocator, being handed to the next request.

**Upstream does not express that as an allocator event.** `ap_proxy_release_connection`
→ `connection_cleanup` (`proxy_util.c`) puts the backend connection on the
worker's reslist and leaves its bucket allocator -- created once per backend
connection in `ap_proxy_connection_create` on `conn->scpool` -- exactly as it
is. The rule that nothing of the old request may outlive the handback is a
copying discipline: `ap_proxy_buckets_lifetime_transform` moves the buckets to
the frontend's allocator, and the brigade is cleaned; the defect is that the
cleanup came after the release. Nothing is freed at the handback, which is
what makes this the corpus's one **reuse-not-free** case (taxonomy class 3).

**The reduced consumer expresses it.** Under the Sublet discipline a lender
that reuses without freeing revokes at the point of reuse, and the reduced
consumer models the lender. So at the handback it ends the allocator's
tenancy -- `apr_bucket_alloc_destroy` and a fresh `apr_bucket_alloc_create`
on the connection pool, the operation `connection_cleanup` would perform --
in both arms; the fix differential stays the cleanup before it. This is the
one step the reduced sequence takes that upstream does not, and the reason
the sublet arm can catch a defect no free-triggered mechanism can see.

The storage is **nodes from the connection's bucket allocator, still named by the brigade**.

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
