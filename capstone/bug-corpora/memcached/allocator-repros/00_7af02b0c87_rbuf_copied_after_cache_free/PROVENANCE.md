# Provenance

**Tier: LITERAL-traceable allocator, reduced consumer.** `cache.c` is
upstream's file at the port's 1.6.45 pin, compiled through
[`ports/memcached/allocators`](../../../../ports/memcached/allocators/README.md):
two patches, one replacing its includes with the port's shim, one connecting
the object transitions to the port's adapter. Every allocation decision is
upstream's. Reduced: the connection struct, the text parser that decided a
switch to `malloc` was needed, and the worker thread that owns the cache.

- **Fix:** `7af02b0c87` — *"core: fix use-after-free for text multigets"*,
  2022-01-12, `memcached.c` +1/−1. The two lines of `rbuf_switch_to_malloc`
  are reordered: copy first, free afterwards. The message: *"Reported in #849
  — this fixes copying a read buffer after freeing the original read buffer.
  This didn't matter for years since the cache code didn't touch the buffer,
  but recently it can reuse the first 8 bytes as a pointer to the internal
  freelist. Thus in some situations where large reads happen the command can
  get corrupted, returning an unhelpful "ERROR" to the end user."*
- **File:** `memcached.c`, `rbuf_switch_to_malloc()` (lines 436–450 at the
  parent commit).
- **CVE:** `NO VERIFIED CVE`. The commit carries none and no advisory
  database was searched for this entry.
- **Live at the pin:** no. GitHub's compare of `1.6.45...7af02b0c87` reports
  `status=behind, behind_by=566` — the fix is an ancestor of the tag. This
  case reconstructs the pre-fix order the commit's own diff shows.

## The defect

Before the fix:

    do_cache_free(c->thread->rbuf_cache, c->rbuf);
    memcpy(tmp, c->rcurr, c->rbytes);

`do_cache_free` pushes the buffer on the cache's `STAILQ`, which stores the
list link in the object's first bytes (`struct cache_free_s` is overlaid on the
freed object). The `memcpy` then copies the unparsed command out of that
buffer — link included.

## What the two arms show, and what they do not

`accessed_through_stale=1 damage=1` in the buggy arm: the copied bytes differ
from the command that was read, because the free list's link now sits at the
front. `damage=0` in the fixed arm. `unit_reissued=1` in **both** arms is the
mechanism, not a defect of the control: the next connection's `cache_alloc`
pops the buffer either way. The case holds a second connection's buffer and
returns it just before the defect, so the free list is not empty at the push
and the link written into the defect's buffer is a real pointer, as in
upstream's description, and not `NULL`; a `CHECK` refuses the fixture if the
two buffers turn out to be one. (The first version of this case freed the
second buffer before taking the first and so tested with an empty list: the
report's `object_reuses=2` against one object carved gave it away.)

What the fixture does not show: the parser going on to misread the corrupted
command and answer `ERROR`. Upstream reports that; this records the bytes
that cause it.

## Why the storage is not visible to a malloc-level tool

`do_cache_free` (`cache.c:135`) calls `free()` only when the cache is over its
limit; otherwise the object goes on the `STAILQ` and `do_cache_alloc`
(`cache.c:80`) pops it, uncleared, for the next request. The rbuf cache's
limit is set from `read_buf_mem_limit`, zero by default, so on this path no
`free()` occurs and same-address reuse is a property of the allocator rather
than of a run — the same argument as APR's node free list.
