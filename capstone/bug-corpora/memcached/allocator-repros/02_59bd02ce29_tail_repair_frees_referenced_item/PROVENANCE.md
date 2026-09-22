# Provenance

**Tier: LITERAL-traceable allocator, reduced consumer — and LIVE IN THE PIN.**
`slabs.c` is upstream's file at the port's 1.6.45 pin, compiled through
[`ports/memcached/allocators`](../../../../ports/memcached/allocators/README.md):
two patches, one replacing its includes with the port's shims, one connecting
the free-list transitions to the port's adapter. Every allocation decision is
upstream's. Reduced: the item layer, to the three functions that decide whether
a chunk goes back to slabs.

- **Upstream state:** unfixed. The branch is in the pinned tree at
  `items.c:1117-1125`, under upstream's own comment:

      /* In case of refcount leaks, enable for quick workaround. */
      /* WARNING: This can cause terrible corruption */
      if (settings.tail_repair_time &&
              search->time + settings.tail_repair_time < current_time) {
          itemstats[id].tailrepairs++;
          search->refcount = 1;
          /* This will call item_remove -> item_free since refcnt is 1 */
          STORAGE_delete(ext_storage, search);
          do_item_unlink_nolock(search, hv);
          item_trylock_unlock(hold_lock);
          continue;
      }

- **Reachable as shipped:** `-o tail_repair_time=N` with `N >= 10`
  (`memcached.c:5291-5298`); the default is 0 (`TAIL_REPAIR_TIME_DEFAULT`,
  `memcached.h:127`), which disables it.
- **The change this case pairs against:** `59bd02ce29` — *"disable
  tail_repair_time by default"*, 2014-04-27: *"We've cleaned up a lot of the
  refcount leaks lately, and this feature does have a race for soemthing
  grabbing active data from the tail."* That is the fix in the sense the
  corpus uses: the shipped default stopped taking the branch. The code was not
  removed, and in 1.6.45 it is still there.
- **CVE:** `NO VERIFIED CVE`. No advisory database was searched for this
  entry; it is upstream's own documented hazard, not a reported vulnerability.
- **Live at the pin:** **yes** — the first case in this corpus for which that
  is true. The other two reconstruct a shape that upstream has since fixed;
  this one runs the code that is in the pinned tree.

## The defect

`do_item_alloc` walks to the LRU tail looking for a chunk to reuse. Its probe
takes a reference of its own, so `refcount_incr(search) != 2` means some other
client is still holding the item (`items.c:1111`). The pathological case is
supposed to be skipped. With `tail_repair_time` set and the item old enough, it
is not skipped: `search->refcount = 1` **overwrites** the count, discarding
every outstanding reference, and `do_item_unlink_nolock` then drops it to zero
— `item_free`, `slabs_free`, and the chunk is on its class's `slots` list while
a client still holds a pointer into it. The next `do_slabs_alloc` for that
class pops it straight back.

## Why the reduction is the defect and not a model of it

Three functions decide whether a chunk reaches the allocator, and all three
are kept with their arithmetic intact:

| upstream | here |
|---|---|
| `items.c:item_free` | `item_free_reduced` — `ITEM_clsid`, then `slabs_free` |
| `items.c:do_item_remove` | `do_item_remove_reduced` — `--refcount`, free at zero |
| `items.c:do_item_unlink_nolock` | `do_item_unlink_nolock_reduced` — clear `ITEM_LINKED`, then remove |

Left out: the hash table, the LRU queues, the stats, `STORAGE_delete` and the
item locks. None of them takes part in the decision, and none of them changes
which chunk `slabs_alloc` returns next. The refcount transitions the case
performs are upstream's: 1 from `do_slabs_alloc`, +1 for being linked, −1 when
the storing client lets go, +1 for the holder, +1 for the eviction probe.

## What the two arms show, and what they do not

`unit_reissued` differs between the arms, which is a deliberate departure from
cases 0 and 1. There the reuse belonged to the allocator and was present in
both arms; here the premature free **is** the defect, so the control cannot
show the reuse it enables. The case therefore checks `unit_reissued` before
publishing the marker, so a protected arm's fault cannot be credited unless
the chunk really did come back.

What the fixture does not show: the crash upstream's comment implies. It
records the step before it — a client reading another item's bytes where it
stored its own.

## Why the storage is not visible to a malloc-level tool

`item_free` ends in `slabs_free` (`items.c:360`), and `do_slabs_free`
(`slabs.c:501`) pushes the chunk onto `slabclass[id].slots`. A page leaves a
class only through the mover, and a chunk never reaches `free()` at all. So
same-address reuse is a property of the allocator rather than of a run — the
same argument as APR's node free list and cache.c's STAILQ.
