# Provenance

**Tier: LITERAL-traceable allocator, reduced consumer.** `cache.c` is
upstream's file at the port's 1.6.45 pin, compiled through
[`ports/memcached/allocators`](../../../../ports/memcached/allocators/README.md).
Every allocation decision is upstream's. Reduced: the proxy backend, the
event thread that owns it, the worker thread that owns the io cache, and the
Lua-facing response object each IO points at.

- **Fix:** `0ad4de66ae` — *"proxy: fix race crash from io obj
  use-after-free"*, 2022-07-25, `proxy_network.c` +12/−2. The message:
  *"obvious once tracked down: STAILQ_FOREACH is actively using io->io_next,
  and the return call can free and potentially reuse the object if the thread
  gets suspended. There're _SAFE forms of the FOREACH but the reset flow
  shouldn't happen often enough to be worth using more than this standard
  pattern."*
- **File:** `proxy_network.c`, `_reset_bad_backend()`; the same commit also
  NULLs two list links elsewhere "out of paranoia".
- **CVE:** `NO VERIFIED CVE`. The commit carries none and no advisory
  database was searched for this entry.
- **Live at the pin:** no. GitHub's compare of `1.6.45...0ad4de66ae` reports
  `status=behind, behind_by=493`. The case reconstructs the loop the diff
  replaces.

## The defect

Before the fix:

    STAILQ_FOREACH(io, &be->io_head, io_next) {
        io->client_resp->status = MCMC_ERR;
        return_io_pending((io_pending_t *)io);
    }

`STAILQ_FOREACH`'s step is `io = io->io_next.stqe_next`, evaluated after the
body. `return_io_pending` hands the IO back to its worker thread, which
returns the object to its io cache — `do_cache_free` pushes it on the cache's
`STAILQ` — and may pop it again for the next request before the event thread
gets to its step. After the fix:

    while (!STAILQ_EMPTY(&be->io_head)) {
        io = STAILQ_FIRST(&be->io_head);
        STAILQ_REMOVE_HEAD(&be->io_head, io_next);
        ...
        return_io_pending((io_pending_t *)io);
    }

The IO is off the list before it is returned, so nothing reads it afterwards.

## Reduction of a race to a sequence

The upstream defect needs two threads and a suspension between them. The
fixture performs the interleaving the commit describes in program order: the
loop body returns the IO, the worker frees it and immediately takes it again
for another request (`unit_reissued`), zeroes it as a fresh request would,
and only then does the loop read `io->io_next`. That read is the probe.

Natively the reissued object's link reads as `NULL`, so the walk stops after
one of three pending IOs (`damage=1`): two requests never receive their error
status. Upstream saw a crash instead, because the reused object's link is
whatever its new owner stored there. The consequence differs; the stale read
does not.

`unit_reissued=1` in **both** arms is the mechanism, not a defect of the
control: the worker pops the object either way.

## Why the storage is not visible to a malloc-level tool

`do_cache_free` (`cache.c:135`) calls `free()` only over the cache's limit;
the io cache has none. On this path the object goes on the `STAILQ` and
`do_cache_alloc` (`cache.c:80`) pops it, uncleared, for the next request. No
`free()` occurs, so same-address reuse is a property of the allocator rather
than of a run.
