# The Sublet port of SQLite's allocators

What is in this directory protects; what makes SQLite compile and run on Capstone is
elsewhere in `ports/sqlite/` and never reads a file from here. The unprotected arm of every
measurement is that build. `SPEEDTEST1_SUBLET=1` in `run-sqlite-speedtest1.sh`, or
`SQLITE_SUBLET_PATCH` in `build-sqlite-capstone.sh`, applies this port on top of it.

| File | What it is |
|---|---|
| `sublet.h` | the primitives, as operations on capability slots: split, take, give, handle, carve, move. No linear capability ever sits in a C variable, so a copy the compiler makes cannot move one away |
| `sublet-3530300.patch` | the port of memsys5 and the lookaside pool: 28 hunks against `sqlite3-capstone.c` as the build produces it from the amalgamation, each hunk classed in the header (I interface, H hierarchy, M metadata layout). Applied with `patch -F0 -p1` to the copy in the build directory, before any instrument's patch |

The primitives are not SQLite's. When the second allocator is ported they move to a shared
home and only the patch stays here.

## The recipe, as it lands in the two allocators

| Operation | memsys5 | lookaside |
|---|---|---|
| block from the level below | the host's region, linear, into a slot (`sqlite3_sublet_grant`) | `sqlite3MallocLinear`: memsys5 keeps the handle, the pool gets the block linear |
| carve | `sublet_carve` at init, `sublet_split` at a split, after `sublet_handle` on the parent | `sublet_carve`, one slot at a time, front to back |
| hand out | `sublet_take`: mrev, delin | the same, on the slot's entry |
| free | `sublet_give`: revoke, the block is linear again | the same |
| merge | `sublet_give_to` on the parent's handle: one revoke for both halves | |
| destroy | | memsys5's free of the pool block: one revoke, every slot dies |
| metadata out of freed memory | links, control bytes, one capability per atom and the parents' handles in the configured heap | one entry per slot beside the pool: link and capability |

Under the port memsys5's pool is one linear capability carved into blocks, one revocation
node each; a split takes a handle senior to both halves first, so a merge is one revoke;
every hand-out is mrev and delin, every free a revoke. The lookaside's block comes from
memsys5 linear, is carved into slots the same way, and dies with one revoke when memsys5
frees it. The allocators' policies are unchanged: the same blocks split and merge, the same
slots pop and push, so the free-to-reuse gaps of A1 are the same under the port, plus one
memsys5 allocation per connection for the lookaside's side table.

## The bookkeeping of A7

From the patch, comments included, blank lines not: 145 lines added, 43 removed, in 28 hunks.

| Class | memsys5 | lookaside | shared |
|---|---|---|---|
| interface | 27 | 10 | 1 |
| hierarchy | 39 | 17 | |
| metadata layout | 48 | 46 | |
| all | 114 | 73 | 1 |

Application code above the allocators: zero lines. memsys5's seven `sqlite3_mem_methods`
keep their signatures, one entry is added between the levels (`sqlite3MallocLinear`), and
three exported functions hand the pool over and read the counters.

Outside the patch, and counted apart from it because they are not SQLite: the domain
adapter `speedtest1_domain.c` has 67 lines in six `#ifdef SPEEDTEST1_SUBLET` blocks, the
glue that takes the pool linear from the host, carves memsys5's tables beside it and prints
the primitive counts; and `sqlite_host.c --arena` creates that pool as a linear region under
a handle the monitor keeps (`REV_BORROWED`).

Given up: a lookaside buffer the application supplies (`SQLITE_DBCONFIG_LOOKASIDE` with a
buffer) is refused, it is not a linear block this level can carve; and the `SQLITE_DEBUG`
poison writes into a freed block are gone, the block is revoked memory.

## What the port relies on, and where that is written

The port's `init` after a revoke assumes the region comes back with its cursor at the end,
which the emulator the passes ran on did and the specification does not: it wants the block
written through first, and the emulator's merge line has since moved to that (Q-07). Under
it the port halts at its first merge with cause 29 until it writes the block through before
`init`. That, the linearity rule the review found (`sublet_take_linear` reads the base before
the store), and the rest of the limits are in the port README's section on speedtest1 and the
Sublet port.
