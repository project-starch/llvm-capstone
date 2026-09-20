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

## The defect
`wd_worker()` created a pool for each iteration, destroyed it, and left the
variable set. A later `if (!ctx)` therefore saw a live-looking handle and the
loop reused a destroyed pool.

    arm=fixed pool_struct_reissued=1 allocated_through_stale=0 other_pool_corrupted=0 freed_to_malloc=0
    arm=buggy pool_struct_reissued=1 allocated_through_stale=1 other_pool_corrupted=0 freed_to_malloc=0

## What the two arms show, and what they do not

`pool_struct_reissued=1` in **both** arms is the result, not a defect of the
control: the reissue is a property of the allocator, not of the bug. APR hands
the destroyed pool's own node straight back to the next `apr_pool_create`, so
the stale handle does not merely dangle — it comes to name a live, unrelated
pool. The arms differ in one thing, whether the consumer still holds that
handle.

`other_pool_corrupted=0` is honest and deliberate. This fixture demonstrates
that allocation proceeds **through** the stale handle into another pool's arena;
it does not demonstrate that the other pool's existing bytes were overwritten,
because whether the two allocations collide depends on sizes and ordering.
Upstream reports crashes; this records the step that precedes them.

## A shape the other corpora do not have

In CPython, PostgreSQL and FFmpeg the thing that dangles is a pointer to an
**object**. Here it is a pointer to the **allocator** — and because the
allocator recycles its own control structure, a stale allocator handle is
silently rebound to a live one. Two parts of the program then believe they own
separate pools that are the same pool.

## Why the storage is not visible to a malloc-level tool

`apr_pool_destroy` ends with `allocator_free(allocator, active)`
(`apr_pools.c:1014`), and `allocator_free` (`:414`) pushes each node onto
`allocator->free[index]`, a size-bucketed LIFO list. Only nodes beyond
`max_free_index` reach the `freelist` that is actually released, and the default
is `APR_ALLOCATOR_MAX_FREE_UNLIMITED`, under which none are. `allocator_alloc`
pops the same node back. So no `free()` occurs, and same-address reuse is a
property of the allocator rather than of a run — the same argument as
PostgreSQL's Slab and FFmpeg's buffer pool.
