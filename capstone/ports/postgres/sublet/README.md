# The Sublet port of PostgreSQL's memory contexts

**Design, and now partly measurement.** The level below is written and runs: it
is `../port/freestanding/pg_subpool.c`, and `../tools/subpool_domain.c` asks
the hardware whether it keeps its promise. Nineteen claims, every one of them
held, and the section "What the hardware said" has them and the three things
the run taught that no amount of reading would have. The manager itself is not
patched yet. It is written before the patch because the patch has one decision
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
the four events that cost more. The two other questions the design had are
settled below and neither costs the claim: the keeper block needs the context
header to leave the sub-pool, and in-place `realloc` of a large chunk becomes a
copy, which the recording takes once per process.

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
| context create | the level below carves a sub-pool of 64 KiB, keeps a handle for it in its own table, and gives the context its header out of memory outside the sub-pool, because a revocation must not reach the header |
| the keeper block | the first `sublet_carve` from the sub-pool, and its capability goes in the context's keeper field rather than being derived from the header's |
| block from the level below | `sublet_carve` from the context's sub-pool slot: the block, linear, in the context's block slot. The sub-pool's handle is senior to it and stays with the level below |
| carve a chunk | `sublet_carve` from the block slot to the chunk's slot, front to back, which is the bump `aset.c` already does |
| hand a chunk out | `sublet_take` on the chunk's slot: `mrev`, then `delin`, so the program gets a copyable alias and the slot keeps the handle |
| free a chunk | `sublet_give` on the chunk's slot: the alias the program still holds dies, and the slot holds the chunk again, linear, for the next `sublet_take` |
| reset | `sublet_give_to` on the sub-pool's handle, keeper included, then carve the keeper again: one revocation, and none at all for a sub-pool never carved |
| delete | the same revocation, and then the level below takes the sub-pool back and returns the header |
| realloc, same size class | nothing: the chunk's region and the capability handed out do not change |
| realloc that moves | carve, copy with the tag-preserving `memcpy`, then `sublet_give` the old chunk |

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

## The keeper block, decided

**The context header leaves the sub-pool, and the keeper block stays in it.**

`aset.c` makes the decision itself, in one macro:

    #define KeeperBlock(set) \
        ((AllocBlock) (((char *) set) + MAXALIGN(sizeof(AllocSetContext))))

The header and the first block are one `malloc`, and the keeper's address is
the header's plus a constant. A reset keeps the keeper "since it shares a
malloc chunk with the context header", in the comment's own words.

That shape cannot survive one revocation, whichever way it is taken. If the
sub-pool covers the header, then revoking it destroys the `AllocSetContext` the
manager is executing on. If the sub-pool spares the keeper so that the header
survives, then every object in the keeper block survives the reset with a live
capability, and the reset no longer means what it says. Only one arrangement is
both safe and one revocation: the header sits in the level below's own memory,
outside anything the handle covers, and the sub-pool holds every block
including the keeper. A reset revokes the sub-pool, keeper included, and the
manager carves the keeper again immediately.

Splitting the sub-pool so that the header is the front of it does not rescue
the arithmetic. A capability is bounded, so `set + MAXALIGN(sizeof(...))` is
exactly the end of a header-sized capability and faults on the first
dereference, adjacent region or not. The keeper has to be reached through a
capability of its own rather than derived from the header's, which is a field:

| site in `aset.c` | what the port does |
|---|---|
| `AllocSetContext` | one field, the keeper's capability |
| `KeeperBlock`, `IsKeeperBlock` | read the field instead of doing arithmetic |
| create | the header from the level below, the keeper carved from the sub-pool, `endptr` from the block rather than from the header |
| reset | carve and initialise the keeper after the revoke, where it re-pointed a surviving block before |
| delete | two returns rather than one free |
| the `keepersize` assertion | a subtraction across two capabilities, and it exists only under `MEMORY_CONTEXT_CHECKING` |

Fifteen to twenty lines, and every one of them is the same fact: a capability
cannot reach out of the object it was given for. That is the change A7 counts,
and it is a change to addressing and not to policy. The block sizes, the
doubling, the free lists, the keeper's role and the order of the chunks are all
untouched.

## `realloc` in place, decided and priced

**It becomes a copy, and the recording says it happens once per process.**

`AllocSetRealloc` has three paths and Sublet keeps two of them unchanged. A new
size in the same class returns the same pointer and does nothing, so the
chunk's region and the capability handed out are untouched and the path is free.
A new size in a different class already allocates, copies and frees, so Sublet
adds a revocation and nothing else.

The third path is a chunk above the context's `allocChunkLimit`, which owns its
whole block: the manager grows it with `realloc()` on the block. A region can
be split and not joined, so no primitive grows one, and the port has to carve a
new block, copy, and revoke the old. That is a policy change and it is reported
as one.

`experiments/a11/postgres/reallocs.py` counts how often the recording takes it.
Of 1 228 019 allocations on the tpcb rung there are two reallocs, and one of
them is external: the timezone parser at startup, copying 5 120 bytes. The
readonly rung is the same two. So the path the port cannot keep costs one
`memcpy` of five kilobytes per process start, and the paper reports it as a
change rather than as a cost.

## The hand-out keeps its handle, decided and priced

**One revocation node an object, and the board arm is a slice.**

A hand-out is `sublet_take`: an `mrev` for the handle the slot keeps, then a
`delin` so the program gets a copyable alias. The `mrev` costs a node, and over
a rung the hand-outs are half the node bill. The alternative is a hand-out that
only delinearises: no handle, no node, and nothing can revoke the chunk until
its context is reset. Since the node pool is what bounds a run on the board,
that alternative is worth pricing rather than dismissing.

`experiments/a11/postgres/deaths.py` follows every object in the recording to
its death:

| how an object dies | tpcb | readonly |
|---|---:|---:|
| reset under it | 73.04% | 72.56% |
| deleted under it | 10.80% | 7.96% |
| freed by a `pfree` | 15.99% | 18.61% |

So the cheaper hand-out would still catch five objects in six, at their
context's teardown, and would give up the sixth.

It is the wrong trade, and the same script says why. The median freed object
lives **one allocation** of its own context, and the ninetieth percentile is
twenty-two. A chunk freed and immediately reused means a capability that
outlived its `pfree` reaches another live object rather than free space, which
is the form of the bug that corrupts rather than crashes. The frees are also
concentrated where it matters: `TopTransactionContext` and `TopPortalContext`
free everything they allocate, and `ExecutorState` a third of it.

The hand-out therefore keeps its handle. The node bill stays at about two an
object, the board arm of A11 runs a slice of the rung whose length the pool
sets, and the paper reports that rather than engineering it away.

## What the hardware said

`../build-subpool-test.sh` and `../run-subpool-test.sh` build the level below
into a domain of its own and run nineteen claims inside it. Every one held. The
image carries no PostgreSQL source, which is the point: a fault inside a
patched `aset.c` is hard to read back to its cause, and this one narrows the
ground to one file.

The claims that matter:

| claim | what held |
|---|---|
| a reset is one revocation | one, counted by the primitive itself |
| the sub-pool works after the reset | a block and sixteen chunks, written and read back |
| a reset with nothing carved is no revocation | none |
| a context that filled its sub-pool | its reset is two revocations and no more |
| a delete of such a context | both sub-pools come back |
| two thousand creates and deletes | no sub-pool and no entry left behind |

The second row is the one only the hardware could answer. A revoke that killed
a linear child hands the region back uninitialised and refuses `init` until it
has been written through, so a sub-pool that carves and hands out again after a
reset is the proof that one revocation per context is implementable rather than
only arguable.

Three things the run taught:

**A region has to be shared linear, and the default is not.** The monitor hands
a domain a shared region either non-linear with a handle kept by the monitor
(`REV_DEFAULT`) or linear with a handle kept by the monitor (`REV_BORROWED`).
The replay's host shared all four regions the default way, because the
unprotected level below walks its arena with ordinary pointer arithmetic and a
linear capability copied by ordinary C code is what the hardware refuses. A
region that is not linear cannot be split, and a handle senior to it has
nothing to revoke. The first run said so exactly, type 1 where it wanted 0, so
`pg_host.c` grew a `--linear-arena` argument rather than a changed constant.

**The share has to be taken in the register it arrives in.** A linear
capability assigned to a C pointer and read back on a later call comes back
non-linear. The second run said that too, so the arena goes into its slot
inside the share handler and not in the body that follows.

**The revocation-node pool is what bounds a run on the board, not cycles.** A
node is allocated by every `split` and every `mrev`, from a bump head with no
reclamation, so a run spends every node it ever made and a revoke returns none.
The test spent 20 105 nodes for 8 040 allocations, which is two and a half
each, and the emulator prints a watermark at the crossing of the old 10-bit
head because on silicon a pool that wraps reuses live ids and the next
capability store blocks with no timeout and no trap.

`experiments/a11/postgres/nodes.py` prices the whole rung from the trace:

| rung | nodes the port would spend | what 65 536 covers |
|---|---:|---:|
| tpcb | 2 446 088 | 2.77% of the trace |
| readonly | 442 542 | 15.00% of the trace |

The hand-outs and the carves are 95% of that bill, at one node each and one and
a quarter million of them. So the board arm of A11 is a slice, and the slice is
set by the node pool rather than by time. Either the run is cut to that length
and says so, or the pool grows. The paper reports it either way, because it is
a property of the hardware and not of this port: two nodes an object is what a
per-object capability discipline costs, and the pool is the ceiling on how many
objects one boot can see.

**One more, about the toolchain.** The fork's clang asserts in
`Value::stripAndAccumulateConstantOffsets` when it emits debug info for
optimised code on this target. `-O1`, `-O2` and `-Og` all crash with `-g` and
all pass without it. Every build script here uses `-O0`, which is why it had not
been met, and it is named because a cycle count at `-O0` is not a cycle count
and the measurement has to say which it used.

## What is open

The patch to `aset.c` itself, which is the next piece, and one question the
patch will answer rather than the design.

- **Whether the chunk header can go back to eight bytes.** Nothing in a chunk
  holds a capability once the free-list link is an index, so the sixteen bytes
  the capability build forced may not be needed under the port. It would mean
  the protected build uses less memory per chunk than the unprotected one.
