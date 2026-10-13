# Wireshark 4.6.8: wmem allocators

Wireshark's memory manager `wmem` -- its core and all four allocators -- built
outside Wireshark, with a replay of allocator traces, a scope model of the
dissection loop, a link example and the build seam the
[wmem defect corpus](../../../bug-corpora/wireshark/wmem-repros) uses. The pin
is the 4.6.8 release archive, verified by SHA-256 in `upstream.json`; both
block allocators are blob-identical from 4.4.0 through 4.6.8.

Why this allocator: in a default Wireshark run, five pools are backed by a
nested allocator that retains its blocks across a reset and reissues their
storage. The per-dissection packet pool (`block_fast`) keeps its first 2 MiB
block and rewinds; the file and epan scopes (`block`) keep every block and
rebuild their free lists. Upstream finds the resulting stale-pointer defects
only by substituting a per-object allocator (`WIRESHARK_DEBUG_WMEM_OVERRIDE`)
so that Valgrind or ASan can see them. This port keeps the production
allocators and makes their objects lifetimes instead.

## Targets

| preset | what it builds |
|---|---|
| `capstone-application` | the replay, the example and the corpus cases as Capstone processes on the virtual profile (`CAPSTONE_SDK`); `-DWM_SUBLET=ON` adds the Sublet protection |
| `cheribsd` | the same for stock CheriBSD purecap ([host/cheribsd](host/cheribsd)), with the supervisor the corpus's CheriBSD arm runs under |
| `native` | the same, wmem as released, and the tests |

Every target builds the `Wireshark::Wmem` library: the six upstream units
(`wmem_core.c`, `wmem_user_cb.c` and the four allocators), the GLib shim in
`src/shared/shim/`, and the scope model in `src/shared/scopes.c`. `g_malloc`,
`g_free` and `g_realloc` are the process's `malloc`, `free` and `realloc`
(`src/shared/system.c`), as a stock build gets them from GLib: libc natively and
on CheriBSD, virtual mallocng on Capstone.

## The patch

`cmake/prepare-source.py` verifies the archive, extracts the wmem subtree and
prepares one of two variants: `reference`, wmem as released, and `sublet`, with
`patches/wireshark-4.6.8-0001-wmem-sublet-lifetimes.patch` applied. Only
`capstone-application` with `WM_SUBLET=ON` builds `sublet`.

Patch 0001 uses the two instructions directly. `CDERIVE` makes a child
capability of a non-linear parent, bounded to a sub-range and carrying MANAGE
over its own children; `CREVOKE(parent, child)` ends a direct child and
everything below it, and faults when the child is not a live direct child.

* Every block `block` and `block_fast` take from the system allocator, jumbo
  blocks included, carries two lifetimes in its header: `life`, derived from the
  malloc'd object, and the `generation` below it (`wmem_allocator.h`).
* Each allocator's consumer functions stay upstream's and work on the
  allocator's own pointers. Three wrappers are registered in their place: alloc
  derives the caller's pointer from the generation of the block that holds the
  chunk, bounded to the request; free revokes it and hands upstream's free the
  allocator's pointer at the same address; realloc does both. The block is found
  by address in the allocator's block lists.
* Freeing a block revokes the system allocator's lifetime of it, and every object
  lies below. A reset that keeps a block -- `block` keeps all of them,
  `block_fast` its first -- revokes the block's generation and derives a new one.
* `block_fast`'s free stays a no-op: an object lives until the reset, or until
  realloc replaces it.
* `block` keeps its free-list links inside free chunks, with the whole block's
  bounds. A chunk handed out is cleared of them over the caller's range, and so
  is the range a realloc grows into.

`simple` and `strict` are untouched: every object of theirs is its own
malloc'd object, which the system allocator bounds and retires itself.

The caller's pointer covers its request and nothing else, neither the chunk
header nor the block header that holds the lifetimes, so a live pointer cannot
reach its block's generation. The cost of finding the block by address is a
walk of the allocator's block list on every alloc, free and realloc.

## Build

From the repository root, after sourcing `capstone/tests/capstone-test-env.sh`:

```sh
cmake --preset native -S capstone/ports/wireshark/wmem
cmake --build /tmp/capstone/wireshark-wmem/build/native
ctest --test-dir /tmp/capstone/wireshark-wmem/build/native --output-on-failure
cmake --preset capstone-application -S capstone/ports/wireshark/wmem \
  -DCAPSTONE_SDK=<virtual SDK> [-DWM_SUBLET=ON] [-DWM_CORPUS_DIR=<corpus>]
cmake --build /tmp/capstone/wireshark-wmem/build/capstone-application
```

The native suite checks the shared trace adapters (`port-support`), replays a
directed trace of 1,661 events across all four allocators -- allocation,
resize, individual free, reset, collection and destruction, jumbo objects
included -- against its expected counts, rejects ten malformed traces, verifies
that the patch series applies strictly and only once, and runs the link example.
The protected variant is exercised on the virtual platform by the corpus's
`virtual-nested-pools` arm.

## History

Until 2026-10-11 this component also carried guarded authority hooks (patch
0001), a per-chunk port of the block allocator over linear Sublet regions
(patch 0002, `src/allocators/sublet`, with its pre-registrations and findings),
a freestanding Capstone domain target with its region backing, security
fixtures and linux-guest loader, and the QEMU runners for them. `CDERIVE` and
`CREVOKE` made the region backing and the chunk port unnecessary, and the
domain target is not part of the virtual platform. Their recorded results stay
in `results/`.
