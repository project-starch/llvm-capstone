# Why per-chunk revocation forces the metadata move — checked before porting

Recorded during the work registered in `PREREGISTRATION-chunk-port.md`, because two of these
findings change what the port is and one of them shrinks it.

## 1. BLOCK_FAST needs nothing. Scope is `wmem_allocator_block.c` alone.

The pre-registration scoped the port to both block allocators. That was wrong, and the evidence is
in the allocator itself:

- `wmem_block_fast_free` is `/* free is NOP */` (`wmem_allocator_block_fast.c:130-134`). There is no
  per-chunk lifetime to protect, so a per-chunk region cannot catch anything a reset does not.
- Its `realloc` grow path allocates, copies and **does not free** the old chunk, so the old storage
  stays live by design. A per-chunk region would not fault there either.
- The existing hooks patch already gives it `wm_narrow` on every hand-out (spatial bounds) and
  `wm_epoch` on the retained block at a reset (temporal). That is complete for this allocator.

So BLOCK_FAST is already covered, and porting it would add cost and no protection. Scope narrowed.

## 2. The call site already exists, and is a no-op only for Capstone

`src/shared/wmem-port-hooks.h` already declares the hook this port needs, at the right place and
with the right ordering — the 0001 patch calls it *before* the free list reuses the storage:

```c
/* An individual free in the recycler allocator. Sublet lends whole regions
 * and ends no epoch here; a mechanism that acts per chunk can. */
static inline void wm_release(void *p, size_t n) {
#if defined(WM_POISONCAP)
  wm_release_chunk(p, n);
#else
  (void)p; (void)n;            /* Capstone: nothing happens */
#endif
}
```

That is why PoisonCap covers the chunk free and Sublet does not. It is an unimplemented hook, not a
missing mechanism. (The `WM_POISONCAP` backend was removed on 2026-10-10.)

## 3. But implementing it is NOT a small change, and this is the load-bearing finding

The obvious reading — "write a Capstone body for `wm_release` and the gap closes" — does not work,
and the reason is structural rather than incidental. Checked against the primitives, not assumed:

1. **Revocation needs a handle, and a handle needs a region.** `sublet_handle` makes a handle senior
   to *a region*; there is no primitive that revokes a sub-range of one. An object handed out by
   `wm_narrow` is a `shrink` of the block's capability — narrower bounds on the **same node** — so it
   cannot be revoked alone. Revoking what it hangs from is the block's senior handle, which is
   exactly `wm_epoch`, and kills every chunk at once.
2. **Giving a chunk its own region means splitting it out of the block**, and `sublet_carve` /
   `sublet_split` *remove* the carved range from the parent: `sublet_carve` moves the prefix to the
   destination and leaves only the remainder in `from` (`capstone/sublet/sublet.h`).
3. **wmem addresses its own bookkeeping through that same block capability.** Chunk headers are at
   `(uint8_t*)block + offset`, and the free-list links live *inside the freed chunk's data* —
   `WMEM_GET_FREE(CHUNK)` is `WMEM_CHUNK_TO_DATA(CHUNK)`, with upstream's own comment "this is what
   the 'data' section of a chunk contains if it is free". Once a chunk's range is carved out of the
   block capability, every one of those accesses is outside the bounds the allocator holds.
4. **The freed region does not rejoin the block.** `sublet_give` returns the region into its own
   slot, linear; rejoining two neighbours needs a handle taken before the split that separated them,
   and for a front-carving allocator the common ancestor of two adjacent free chunks also covers the
   live objects between them. micropython's port states the same result for the same reason and
   measured its cost (`ports/micropython/patches/0025-…-no-coalescing.patch`).

**So per-chunk revocation is inseparable from moving wmem's metadata out of the block.** There is no
version of this port that revokes chunks and leaves the free lists where they are.

That is worth stating precisely because it is H4's mechanism — "metadata may no longer live in freed
memory, since freed memory is revoked" — arriving as a *forced consequence* rather than as a
tendency. H4's line-count prediction can still be refuted, as PostgreSQL refuted it; what cannot be
avoided here is the move itself.

## 4. Consequences for the port

- The side tables must carry what the block can no longer hold: the free-list and recycler links,
  the chunk headers, and a slot per chunk holding its region (free) or its handle (out).
- **Coalescing goes**, per (4) above, with the micropython precedent.
- `free_all` is unaffected and stays one revoke per retained block on the senior handle: a handle
  taken before any split covers every chunk below it.

## 5. The baseline is reproduced, so the control is real

The existing port builds here (native, capstone-domain and linux-guest) and its 26 security cells
run under QEMU. `mode=1 case=4` **passes as `completed`** — the documented gap — which is the
matched control this port has to flip to `fault`.
