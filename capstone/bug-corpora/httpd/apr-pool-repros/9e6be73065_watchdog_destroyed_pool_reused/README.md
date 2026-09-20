# mod_watchdog: the handle outlives the pool

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
