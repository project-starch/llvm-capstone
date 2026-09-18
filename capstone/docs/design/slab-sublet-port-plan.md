# Porting slab.c to the Sublet discipline

*Design record, 2026-09-18. What the second allocator port changes, function by
function, against PostgreSQL 17.0's `src/backend/utils/mmgr/slab.c` (1154 lines).
The template is `patches/postgresql-17.0-0004-allocset-sublet-context-revocation.patch`;
the interface is `src/allocators/sublet/pg_subpool.h`.*

## Why slab, and why it is second rather than first

Only AllocSet is ported today. Of the eight PostgreSQL specimens live in the
17.0 pin (`ref/postgres-nested-allocator-defects.md`), seven allocate from an
AllocSet and one — `9e0b4b1ab5`, the reorderbuffer INSERT ON CONFLICT
use-after-free — allocates from a Slab context
(`reorderbuffer.c:329`, `SlabContextCreate(..., sizeof(ReorderBufferChange))`).
Porting slab converts that row from a vulnerable arm into a matched pair.

It is also the allocator that makes a corpus row *deterministic*. AllocSet
reuse depends on size class and on what else the workload allocated; slab has
one chunk size, a LIFO free list and no coalescing, so `SlabFree` followed by
`SlabAlloc` returns the identical address every time
(`slab.c:731-732` ↔ `slab.c:277-286`). A specimen that must reproduce on demand
wants that.

Bump and generation are deliberately not first. Bump hosts **no** specimen — its
only consumers are the TidStore radix tree's internals (`tidstore.c:185`, and
only on the `insert_only` path) and `tuplesort.c:777`, and no defect in the
inventory holds a stale pointer into either. Generation hosts one half of the
reorderbuffer's storage (`reorderbuffer.c:349`, tuple data) but no inventory row.

## What is already done, and does not need redoing

**The chunk header is shared and already ported.** slab uses the same
`MemoryChunk` as aset — `slab.c:514` `MemoryChunkSetHdrMask(chunk, block,
MAXALIGN(slab->chunkSize), MCTX_SLAB_ID)`, and `MemoryChunkGetBlock` at
`:712`, `:835`, `:871`, `:896`, `:1115`. Patches 0002 (capability alignment) and
0003 (out-of-band metadata indices) cover that header for every context type, so
the slab port is only the slab-specific half.

**The header slot fits, with room to spare.** `pg_subpool_header` hands out
`PG_HEADER_BYTES` = 320 per context. Modelled against the capstone64 purecap ABI
(16-byte capabilities, 16-byte aligned):

| | bytes |
|---|---|
| `MemoryContextData` | 144 |
| `SlabContext`, stock | **320 — exactly at the limit, zero headroom** |
| `SlabContext`, ported (block lists as indices, plus a pool capability) | **208** |

So the stock context only just fits and any future field would break it, while
the ported one leaves 112 bytes. Indexing the block lists is required by rule 1
anyway; it also buys the headroom. The port must carry the same
`_Static_assert` the aset patch has, for the same reason.

## The three rules, and where slab violates each

The aset patch states the three facts that force every change. Slab violates all
three, but not in the same proportions.

### Rule 3 — "a revoked chunk cannot be read or written"

This is slab's largest change and the one with no analogue left to copy.

`SlabFree` (`slab.c:730-732`) pushes the freed chunk onto the block's free list
**by writing the link into the chunk being freed**:

```c
*(MemoryChunk **) pointer = block->freehead;
block->freehead = chunk;
```

and `SlabGetNextFreeChunk` (`slab.c:277-286`) reads it back out:

```c
block->freehead = *(MemoryChunk **) SlabChunkGetPointer(chunk);
```

Under the discipline a freed chunk is revoked, so neither access can happen. The
link becomes an index chain in the side table: `pg_chunk.next_free` already
exists for exactly this and is documented as "the manager's size-class list, by
index". Slab has one size class, so it is a single chain per block rather than
per class.

**This is the one place the interface must grow.** The chain needs a head, and
`pg_block` has no field for it — `chunk_head`/`chunk_tail` hold carve order and
are needed for the walk, `freeptr` is the carve cursor, `free_next` is the block
table's own free list. Add one field:

```c
unsigned int free_head;   /* the manager's free chain for this block, by index */
```

Four bytes per block entry, set to 0 (the interface's "none") at carve time.
This is an additive change to `pg_subpool.h` and `context-pools.c` and does not
affect the aset port.

`CLOBBER_FREED_MEMORY` in `SlabFree` (`:739-742`) wipes the chunk's payload
around the link. It goes: the chunk is revoked, so there is nothing to wipe and
nothing that could read it.

### Rule 1 — "a capability cannot reach out of the object it was given for"

Four constructs compute an address they were not handed:

| Construct | Line | Becomes |
|---|---|---|
| `SlabChunkGetPointer(chk)` = `chk + sizeof(MemoryChunk)` | `:158` | the carve's own capability, from `pg_subpool_hand` |
| `SlabBlockGetChunk(slab, block, n)` = `block + hdrsz + n*fullChunkSize` | `:165` | `pg_subpool_carve`, or the entry at an index |
| `SlabChunkIndex` / `SlabChunkMod` — chunk offset by pointer subtraction | `:176`, `:186` | index arithmetic on the entry |
| `block->unused` bumped by `fullChunkSize` (`:298`, `:578`) | | `pg_subpool_carve(b, fullChunkSize)` on demand |

`MemoryChunkGetBlock(chunk)` (`:712` and four others) already resolves through
the ported header's block index — patch 0003's work — so those sites need only
`pg_subpool_block_at`.

### Rule 2 — "a revocation cannot spare part of what it covers"

`SlabBlock` is a six-field header at the front of every block
(`Slab_BLOCKHDRSZ`, `:77`). It must leave the block, exactly as `AllocBlockData`
did. Its fields map onto `pg_block` with no loss:

| `SlabBlock` | `pg_block` |
|---|---|
| `slab` (owning context) | `aset` (the field is named for aset; it holds the owning context) |
| `nfree`, `nunused` | new small fields, or slab-side arrays indexed by block |
| `freehead` | `free_head` (the addition above) |
| `unused` | `freeptr`, the carve cursor, already an integer for this reason |
| `node` (one `dlist_node`) | `prev`/`next`, the manager's block list |

A block is on exactly one list at a time — one of `blocklist[0..2]` or
`emptyblocks` — so a single `prev`/`next` pair suffices; only the four list
*heads* multiply, and those live in the context as indices.

## Function by function

| Function | Line | Change |
|---|---|---|
| `SlabContextCreate` | `:322` | header from `pg_subpool_header`, not malloc; create the sub-pool; `_Static_assert` on the header size |
| `SlabGetNextFreeChunk` | `:271` | pop `free_head` by index; carve instead of bumping `unused` |
| `SlabAllocFromNewBlock` | `:539` | `pg_subpool_block` instead of `malloc(blockSize)`; initialise the side entry |
| `SlabAllocSetupNewChunk` | `:498` | `pg_subpool_hand` to produce the returned capability |
| `SlabAlloc` | `:630` | unchanged in policy; the chunk comes from the two above |
| `SlabFree` | `:701` | `pg_subpool_drop` to revoke; push by index; no write into the chunk |
| `SlabReset` | `:431` | **one `pg_subpool_reset`** replaces both block walks and every `free(block)` |
| `SlabDelete` | `:485` | `pg_subpool_destroy` + `pg_subpool_header_free` |
| `SlabRealloc` | `:826` | block via `pg_subpool_block_at`; the policy needs nothing — it returns the caller's own pointer when `size == chunkSize` (`:851-852`) and raises an error otherwise, so there is no growth path to re-express |
| `SlabGetChunkContext`, `SlabGetChunkSpace` | `:863`, `:887` | block via `pg_subpool_block_at` |
| `SlabIsEmpty`, `SlabStats` | `:912`, `:929` | walk lists by index; never touch a free chunk |
| `SlabCheck` | `:997` | the chunk walk cannot be done — same reasoning as `AllocSetCheck` in the aset patch. `isChunkFree` becomes the bookkeeping check |

`isChunkFree` (`:114`, allocated at `:409-410`) deserves its own note. It is a
per-block bool array that already exists, compiled only under
`MEMORY_CONTEXT_CHECKING`, used by `SlabCheck` to cross-check the free list. It
is precisely the out-of-band free-marking structure rule 3 requires. The port
should promote it to always-on and make it the allocator's real record of which
chunks are free, rather than a debug shadow of a list that can no longer exist.
That reduces the change rather than adding to it.

## Policy that must not change

A7 counts this port's lines against the unprotected arm, so the comparison is
only meaningful if the policy is identical: same chunk size, same
`chunksPerBlock`, same `blockSize`, same three-element blocklist and its shift,
same `SLAB_MAXIMUM_EMPTY_BLOCKS` retention, same LIFO order within a block. As
with `ALLOC_BLOCKHDRSZ` in the aset port, `Slab_BLOCKHDRSZ` keeps its value even
though no header occupies those bytes any more, because it feeds
`chunksPerBlock` and therefore the whole geometry.

## Build wiring

- `src/allocators/sublet/unsupported-allocators.c` — its refusal message names
  slab. Once slab is ported the message must name only generation and bump, and
  the refusal must still fire for those.
- `cmake/prepare-source.py` — add a slab variant beside the aset ones
  (`:113-117` builds the patch lists).
- `cmake/Replay.cmake` — `pg_manager` compiles `slab.c` from `${PG_SOURCE}`;
  the Sublet mode must take the variant instead, as it already does for `aset.c`.
- `ledger.manifest` — one row for the new patch, classified like the aset one.

## Validation this port needs, and what blocks it today

The aset port is validated by building the domain and running the QEMU replay
suites. **Neither can be run on this machine: there is no Capstone clang build.**
Verified rather than assumed — `/usr/bin/clang` is stock LLVM 18 with no
capstone64 target, no `cmake-build*` directory exists anywhere under the home
directory, and `capstone/tests/build-toolchain.sh` requires an existing
`build.ninja` that is not there.

Building it is not a step to take casually: that script takes a **machine-wide
exclusive lock** and runs `ninja -j90` under a 64 GiB cap, because a `-j56` build
landing on another lane's measurement caused the 5-6 September outage. It is a
shared-resource decision, not a local one.

So the order is: build or obtain the toolchain, then implement against this
plan, then validate with

1. domain build of the slab Sublet variant, including the header `_Static_assert`;
2. the QEMU replay suites, spatial and Sublet, byte-identical results to the
   pre-port arm for every aset-only workload — the port must not perturb the
   allocator it is not touching;
3. a slab-allocating workload, which the current recorded trace does not contain:
   every one of the 23,096 and 5,123 contexts in the recorded pgbench rungs is an
   aset. A slab arm needs either a logical-decoding recording or a synthetic
   fixture in `tests/make-fixture-trace.py`, and that is a prerequisite, not an
   afterthought — without it the ported code would never execute and a green run
   would mean nothing.

Point 3 is the one most easily missed. A port with no workload that reaches it
produces a clean result from a check that cannot fire.
