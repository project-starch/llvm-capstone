# Provenance

**Tier: LITERAL-traceable allocator, reduced consumer.** `apr_pools.c` is
upstream's file at the port's 1.7.4 pin, byte for byte, compiled through the
census seam. Reduced: `wd_worker`'s loop, its watchdog callbacks and its
threading, none of which changes which pool is destroyed or which handle
survives.

- **Fix:** `9e6be73065` — *"modules/core/mod_watchdog.c (wd_worker): Fix crashes snuck into r1876599 where a destroyed pool was reused."*, 2020-04-16. The variable is renamed and set to `NULL` after `apr_pool_destroy`.
- **File:** `modules/core/mod_watchdog.c`.
- **CVE:** `NO VERIFIED CVE`. The commit carries none and no advisory database was searched for this entry.
- **Live at the pin:** the APR pin is the allocator's, not httpd's; this corpus does not pin httpd, so the fixture reproduces the call sequence rather than a shipped tree. Stated here rather than implied.

## Why the storage is not visible to a malloc-level tool

`apr_pool_destroy` ends with `allocator_free(allocator, active)`
(`apr_pools.c:1014`), and `allocator_free` (`:414`) pushes each node onto
`allocator->free[index]`, a size-bucketed LIFO list. Only nodes beyond
`max_free_index` reach the `freelist` that is actually released, and the default
is `APR_ALLOCATOR_MAX_FREE_UNLIMITED`, under which none are. `allocator_alloc`
pops the same node back. So no `free()` occurs, and same-address reuse is a
property of the allocator rather than of a run — the same argument as
PostgreSQL's Slab and FFmpeg's buffer pool.
