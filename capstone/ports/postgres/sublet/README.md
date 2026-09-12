# The Sublet port of PostgreSQL's memory contexts

**Design, not measurement, except where it says otherwise.** No line of the
port has run. It is written before the patch because the patch has one decision
in it that decides whether the claim the paper makes is true, and that decision
is better argued than discovered. The one number the design needed, the
sub-pool's size, is measured: the recording of the two pgbench rungs prices it,
and the section that reports it says which script produced it.

What is in this directory protects. What makes the memory manager build and run
under capabilities is the rest of `ports/postgres/` and never reads a file from
here. The unprotected arm of every measurement is that build.

## The claim this has to make true

A11, C11: *a context tree dies on the pgbench trace in one revocation per
context, and the allocator costs X% over spatial.* H1 is the sharp half: the
cycles of a reset grow with the number of child contexts, not with the objects
in them and not with the bytes.

The recording says the design reaches it on 99.96% of the resets and deletes
of the tpcb rung and 99.92% of the readonly rung, and that a third of them need
no revocation at all. The section on the sub-pool's size has the numbers and
the four events that cost more.

So "one revocation per context" is the thing to build, and the obvious port
does not deliver it.

## Why the obvious port does not deliver it

`aset.c` takes its blocks from the level below one at a time, whenever a
context runs out of room, interleaved with every other context doing the same.
A handle from `mrev` is senior to one node, and `revoke` walks the junior run
below it, so a handle the level below keeps for one block covers that block and
nothing else. A context with *k* blocks would then cost *k* revocations, and a
reset's cost would grow with the blocks, which grows with the objects. H1
would be false, and it would be false because of how the port was built rather
than because of anything about the hardware.

## The decision: the level below delegates per context, not per block

The level below gives each context a **sub-pool**: one region, and it keeps one
handle senior to it. Every block that context asks for is carved from its own
sub-pool. A reset or a delete is then `sublet_give_to` on that one handle: one
revocation, and every block and every chunk and every capability the program
still holds into any of them dies with it.

This is a change to the level below and not to the manager. `aset.c`'s policy
is untouched: the same block sizes, the same doubling, the same keeper block,
the same free lists, the same chunks in the same order. What changes is where
the bytes come from, and that is the level the port owns anyway. A7's condition
that the allocator's policy is unchanged is met, and the line count it reports
stays a count of what the manager needed.

It is also the shape `sublet.h` describes in its own words: *a handle taken by
the level below covers a whole sub-pool.* The hierarchy in the hardware follows
the hierarchy in the program, and it is the level below that makes it possible,
by delegating once per context instead of once per block.

A child context is a sub-pool of its own, under its own handle. A parent's
revoke does not reach it, so a tree of *n* contexts costs *n* revocations, in
the order `MemoryContextDelete` already walks it, bottom up. That is what C11
says: one per context. Not one for the tree.

## The recipe, operation by operation

| Operation | What the port does |
|---|---|
| context create | the level below carves a sub-pool from its region, keeps a handle for it in its own table, and hands the context the sub-pool linear |
| block from the level below | `sublet_carve` from the context's sub-pool slot: the block, linear, in the context's block slot. The sub-pool's handle is senior to it and stays with the level below |
| carve a chunk | `sublet_carve` from the block slot to the chunk's slot, front to back, which is the bump `aset.c` already does |
| hand a chunk out | `sublet_take` on the chunk's slot: `mrev`, then `delin`, so the program gets a copyable alias and the slot keeps the handle |
| free a chunk | `sublet_give` on the chunk's slot: the alias the program still holds dies, and the slot holds the chunk again, linear, for the next `sublet_take` |
| reset | `sublet_give_to` on the sub-pool's handle, except the keeper: one revocation, and none at all for a sub-pool never carved |
| delete | the same, keeper included, and the level below takes the sub-pool back |
| realloc that moves | `sublet_give` the old chunk after the copy, and the copy is the tag-preserving one |

## What has to move out of freed memory, and where it goes

This is the class the port bookkeeping expects to dominate, and here it is one
thing with two names.

`aset.c` keeps the free list of a size class **inside the freed chunk**:
`GetFreeListLink(chunk)` is the chunk's own payload, and `link->next` is the
next free chunk of that class. Under Sublet the chunk is revoked when it is
freed, so its bytes are not readable, and the link cannot live there. It is
also the place that forced the sixteen-byte chunk header on the unprotected
build, for a different reason: a capability does not fit in an eight-byte slot.

Both go to a **side table, one entry per chunk**, in memory the level below
gives the port and the program never sees:

| field | what it is |
|---|---|
| `sublet_cap slot` | the chunk's capability: the region while it is free, the handle while it is out |
| `uint32 next_free` | the free list of its size class, by index rather than by pointer |

An entry is sixteen bytes plus four, against a chunk of at least sixteen. The
table is therefore the same order as the heap it describes, which is what the
SQLite port measured too, and it is a cost the paper reports rather than one it
hides: the discipline needs a capability per live object, and a capability is
sixteen bytes.

The free list becomes an index into the table, which also removes the reason
the chunk header grew: nothing in a chunk holds a capability any more. Whether
the header can go back to eight bytes under the port is a question for the
measurement, and it is worth asking, because it would mean the protected build
uses **less** memory per chunk than the unprotected capability build.

## The sub-pool's size, measured

**Sixty-four kilobytes, and a second region for the context that fills it.**

A context whose sub-pool runs out has to be given another, and then it holds
two handles and its reset is two revocations. A larger sub-pool buys fewer of
those, and it is not free: a Capstone region is memory and not address space,
because a domain has no demand paging, so a sub-pool a context never fills
still occupies the arena for as long as the context lives. The size is
therefore a trade of arena against revocations, and the recording prices both
at once. `experiments/a11/postgres/regions.py` in the paper's repository does
the pricing; `subpools.py` beside it reports what a single context needs.

The two pgbench rungs, counted per event rather than per context, because what
a reset costs depends on what that context had asked for since the last one:

| sub-pool | peak arena, tpcb | one revocation, tpcb | peak arena, readonly | one revocation, readonly |
|---|---:|---:|---:|---:|
| 32 KiB | 12.8 MiB | 90.43% | 6.1 MiB | 79.94% |
| 64 KiB | 16.4 MiB | 99.96% | 9.6 MiB | 99.92% |
| 128 KiB | 23.6 MiB | 99.97% | 16.6 MiB | 99.94% |
| 1 MiB | 127.0 MiB | 100% | 118.0 MiB | 100% |

Sixty-four kilobytes is the knee, and one context puts it there. `ExecutorState`
asks for up to 94 880 bytes between resets and resets five thousand times on
the tpcb rung, so at 32 KiB every one of those resets costs two revocations and
the share falls to 90%. Above 64 KiB the arena grows and the share does not:
the four events that still cost more than one revocation are `MessageContext`
on a long query, `TopTransactionContext` once, and an authentication file read
twice at startup. A megabyte would buy them, at eight times the arena.

The arena number is the one that rules out the generous choice. Only 120
contexts are alive at the peak on tpcb and 117 on readonly, out of 23 096 and
5 123 created, so a megabyte each costs 127 MiB where the replay's arena is
64 MiB. Sixty-four kilobytes each costs 16.4 MiB and fits what the host already
gives the domain.

Two further numbers the same pass produced, and both help:

**A third of the resets have nothing to revoke.** On tpcb 9 973 of 30 982 resets
and deletes act on a context that had asked for nothing since the last one, and
on readonly 2 002 of 7 011. `ExprContext` is almost all of it: 8 971 contexts
created and deleted on tpcb without ever holding an object. A sub-pool never
carved needs no revocation, so the port passes these by, and the cost of
tearing down a context tree follows the contexts that held an object rather
than the contexts.

**The side table is small.** At the peak 2 293 chunks are alive on tpcb and
2 116 on readonly, so at twenty-four bytes an entry the table's peak is a tenth
of a megabyte. The per-chunk cost is real and the paper reports it, but it is
not what decides whether the port fits.

## What is open

- **The keeper block.** A reset keeps the first block. One revocation cannot
  spare part of what it covers, so the keeper needs either its own handle,
  which makes a reset two revocations by construction, or to be re-carved and
  re-filled after the revoke, which makes it a write of the keeper. The second
  is what `sublet_give_to` already does to a region it hands back, so it may be
  free.
- **`realloc` in place.** `aset.c` grows a chunk into the space after it when
  that space is free. Under Sublet the chunk's region would have to be extended,
  which no primitive does: a region can be split, not joined, except by
  revoking a handle senior to both halves. The port may have to make that path
  a copy, which is a policy change and has to be reported as one.
