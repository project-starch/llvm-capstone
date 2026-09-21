# memcached 1.6.45 slabs and object cache

The real allocators in `slabs.c` and `cache.c`, extracted into native and
Capstone builds through the seam the [census](../README.md) measured. This is
an allocator component port for the
[memcached bug corpus](../../../bug-corpora/memcached/allocator-repros/README.md).
It does not execute memcached, its item layer, its threads or its page mover:
the port is single-threaded, the mutexes are no-ops in a domain and the host's
uncontended ones natively, and `slabs_mover.c` -- the one place a page changes
class -- is not built. There is no CheriBSD or PoisonCap build of these
allocators.

## Layout and source boundary

This component uses `../../common` for verified downloads, cross toolchains,
external builds and the shared QEMU lock, in the layout the port catalog
describes. `src/shared/` holds the protocol, the metadata heap, the service
stubs and the lifetime ledger; `src/allocators/sublet/` and `src/native/` the
two authority layers under it; `src/capstone-domain/` the domain entry;
`src/linux-guest/` the staging loader. Generated sources stay outside the
repository.

`upstream.json` pins the official memcached 1.6.45 archive by SHA256 -- the
same archive `../fetch-memcached.sh` caches, so one download serves the census
and the port. Two ordered patches, each naming its input, prerequisites and
application command:

| patch | what |
|---|---|
| `0001-freestanding-shims` | `slabs.c` includes `memcached.h` whole -- 1114 lines -- and eleven system headers; `cache.c` four more and `cache.h` `<pthread.h>`. They become the four shims in [`../adapted/`](../adapted/): what `slabs.c` reads from `memcached.h`, each line cited to the 1.6.45 tree; the libc pieces the patched allocator still calls, renamed so a hosted client never sees them; the same for `cache.c`; and the mutexes. Two regions are gated on `MC_PORT`: the stats formatter and huge-page preallocation. Ten hunks in `slabs.c`, one each in `cache.c` and `cache.h` |
| `0002-lifetime-hooks` | connects the free-list transitions to the adapter: ten hunks in `slabs.c`, five in `cache.c` |

## Lifetime mapping

memcached has two units of storage here, and neither ever reaches `free()`
on the path that matters.

A **chunk** is slabs' unit: a 1 MiB page is cut once into `perslab` chunks of
its class's size, all pushed on the class's `slots` list; `do_slabs_alloc`
pops one, `do_slabs_free` pushes it back, and a page leaves a class only
through the mover. An **object** is `cache.c`'s: `malloc`'d once, pushed on a
`STAILQ` at `cache_free` and popped uncleared at `cache_alloc`, freed only over
a limit the three per-thread instances do not have. Upstream still decides
everything about both -- the class table, `perslab`, the LIFO order, which
chunk or object the next request pops. The adapter supplies the storage under
a page, a chunk or an object and, in the protected mode, the authority over it.

The hooks sit on those transitions, not on `free()`. `mcp_chunk_release`
runs where `do_slabs_free` and `do_slabs_free_chunked` file a chunk, after its
header has been read and before its links are written; `mcp_chunk_issue`
where `do_slabs_alloc` pops one; `mcp_object_release` and `mcp_object_issue`
at the two corresponding lines of `cache.c`. Each returns the alias upstream
uses from then on. Pages come from the payload region in place of
`memory_allocate`'s `malloc`; once `do_slabs_newslab` has zeroed a page the
whole-page alias is exchanged for one region per chunk (`mcp_page_carve`), and
the split asks for each chunk by number (`mcp_chunk_at`), because a pointer
stepped past its own chunk is not the next chunk's alias. The slab list grows
element-wise, since it holds page aliases. `cache.c`'s control block and name
come from the metadata heap, so an allocator never lives inside storage it
hands out.

Both domain modes use the same layout and allocator code:

- `spatial`: a chunk or object keeps the alias it was carved with. A pointer
  held across `slabs_free` or `cache_free` still names the storage, which by
  then is the next item's.
- `sublet`: release and issue each `sublet_give` then `sublet_take` the unit,
  so that pointer is a revoked alias. Revocation clears the chunk; the one
  header field upstream keeps on free memory, `slabs_clsid`, is put back from
  the page's record. A chunk filed by the split has never been held and is
  not revoked then.

The ledger, `src/shared/leases.c`, is one file for both targets; only the
authority layer under it differs (`src/allocators/sublet/authority.c`,
`src/native/authority.c`), so the native arms measure the same bookkeeping
the domain does. It keys chunks by page map and offset and objects by a
sorted table, and refuses a release or issue that does not match a unit's
state or class with a 5xx code rather than guessing.

`CHUNK_ALIGN_BYTES` is 16 here where upstream aligns chunk sizes to 8
(`-DCHUNK_ALIGN_BYTES=16`, a knob the shim defaults to 8 so the census keeps
upstream's table). A Sublet region must be aligned and sized to whole
capabilities, and a chunk is a region. `NDEBUG` is memcached's own production
build (`Makefile.am:92`). Native builds use both, so the class table agrees
across the seam: with `sizeof(item)` 48 natively the first class is 96 bytes,
with 80 in a domain (pointers are 16 bytes there) it is 128, and either way
the table is what `slabs_init` computes from upstream's defaults
(`settings_init`, `memcached.c:224-258`), which `src/shared/services.c`
carries.

The fixed regions are 64 MiB payload -- 48 MiB of slab pages from the bottom,
16 MiB of cache objects from the top -- 16 MiB metadata, 8 MiB trace and 4 KiB
report; at most 768 pages and 8,192 objects, and one record per chunk, so a
page of the smallest class costs about 0.5 MiB of metadata in a domain.
`metadata` in the report is heap high-water usage. No timing or
memory-overhead claim is made.

## What is not built, and why it matters

- **The page mover** (`slabs_mover.c`, `slab_automove*.c`). It is the one
  path on which a chunk's storage becomes another class's page, and every
  slabs-level lifetime defect on record in upstream's history is in it. A
  port that hooked it would be porting a second allocator; this one names
  the boundary instead.
- **Threads.** 31 `pthread_mutex` references in `slabs.c` compile to nothing
  in a domain. The LRU maintainer and the mover are threads; neither runs.
- **Stats and preallocation**, gated under `MC_PORT`: the former needs the
  server's per-thread stats aggregate, the latter the host OS.
- **`slabs_init`'s `double` arithmetic** is upstream's and is kept; the
  freestanding target has no FP ABI, so nine compiler-rt double builtins are
  compiled in from the tree, as the BEEBS FP benchmarks do.

## Build

From the repository root, source `capstone/tests/capstone-test-env.sh` and set
`CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR`, `CAPSTONE_QEMU_BINARY` and
`PORT_MUSL_ROOT` to prepared tools. From this directory:

```sh
cmake --preset native && cmake --build /tmp/capstone/memcached-allocators/build/native
ctest --test-dir /tmp/capstone/memcached-allocators/build/native
cmake --preset capstone-domain && cmake --build /tmp/capstone/memcached-allocators/build/capstone-domain
cmake --preset linux-guest && cmake --build /tmp/capstone/memcached-allocators/build/linux-guest
```

Presets build under `/tmp/capstone/memcached-allocators/build/`. A hosted
build exposes `Memcached::Allocators`, `bin/allocator-example` and
`PORT_CLIENT_SOURCE`. The one-source seam is `-DMCP_CORPUS_SRC=<case.c>`: the
corpus supplies `mcp_replay`, and the build produces `bin/defects` (hosted) or
`bin/defects.dom` (domain). The corpus's `shared/build-cases.sh` invokes it
once per case. `allocator-example` prints
`ALLOCATOR_EXAMPLE memcached PASS pointer_bytes=8` as its last line, the form
the shared runner's component registration expects.

## Verification

The native suite checks patch order and duplicate rejection, that preparation
failure preserves a previous complete source, and that `allocator-example`
observes both reissues the corpus exists for: a freed chunk comes back as the
next `slabs_alloc` and a freed object as the next `cache_alloc`,
`chunk_reuses >= 1` and `object_reuses >= 1`. Without that the example would
be measuring the wrong allocators. The same sequence, built through the seam
as a domain, completes in both modes under QEMU with the same counts.

The domain arms are the corpus's, run and judged by
[its runner](../../../bug-corpora/memcached/allocator-repros/runners/capstone-domain/README.md):
`spatial` must complete, `sublet` must fault at the labelled probe, the
expected address is published by the run, and the negative control must make
every oracle fail before a pass is believed. Results are recorded in the case
files' `status` and in the pull request; run artifacts stay under
`/tmp/capstone` and are not committed.
