# Provenance

**Tier: LITERAL-traceable allocator, reduced consumer.** `slabs.c` is
upstream's file at the port's 1.6.45 pin, compiled through
[`ports/memcached/allocators`](../../../../ports/memcached/allocators/README.md).
Every allocation decision is upstream's. Reduced: the item layer, and the race
to the interleaving the fix's own message describes.

- **Fix:** `152ddb68f7` — *"ascii: fix unlocked refcount-- on error path. If a
  too-large opaque token is submitted with an mg, the refcount was decremented
  without the item lock held in some cases. this could cause the refcount to
  drift on a busy item and lead to memory corruption. Inert unless someone is
  constantly sending overlarge opaques."*, 2026-07-09, `proto_parser.c` +3/−1.
- **The diff:**

      error:
          if (it) {
      -       do_item_remove(it);
              if (of.locked) {
      +           do_item_remove(it);
                  item_unlock(hv);
      +       } else {
      +           item_remove(it);
              }
          }

  `do_item_remove` assumes the item lock is held; `item_remove` takes it.
  Before the fix the error path used the first either way.
- **CVE:** `NO VERIFIED CVE`. The commit carries none and no advisory database
  was searched for this entry.
- **Live at the pin:** no, by one day. The fix is 2026-07-09 and the pin is
  1.6.45, tagged 2026-07-10; `process_mget_cmd`'s error path in the pinned
  tree already has both branches.

## The defect

`refcount_decr` is `--(it->refcount)` (`memcached.h:1041`): a read, a subtract
and a store. Under the item lock nothing can land between them. Without it, a
concurrent `do_item_get` on the same item increments between the read and the
store, and its increment is overwritten — the drift the message names. The
count is then one short of the number of holders, and the next release reaches
zero while a client is still holding the item: `item_free`, `slabs_free`, and
the chunk is on its class's free list.

## Reduction of a race to a sequence

The three steps of the decrement are written out with the concurrent get
placed between them, because that interleaving *is* the defect; a single `--`
would hide it. The fixed arm serialises the two operations, which is what
taking the lock achieves. This is the corpus's second race-as-sequence, after
case 1, and it is stated here rather than implied.

Left out: the threads, the lock itself, the `mg` parser and the overlarge
opaque token that leads to the error path. None of them changes the arithmetic
or which chunk `slabs_alloc` returns next.

## What the two arms show

`unit_reissued` differs between the arms, as in cases 2 and 3. The case checks
it before publishing its marker, so a protected arm's fault cannot be credited
unless the chunk really did come back.

What the fixture does not show: that the interleaving occurs in practice.
Upstream says it is *"inert unless someone is constantly sending overlarge
opaques"*. This records what the corruption is when it does, not how likely it
is.
