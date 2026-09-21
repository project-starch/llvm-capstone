# APR 1.7.4 pools

The real allocator in `memory/unix/apr_pools.c`, extracted into native and
Capstone builds through the seam the [census](../census-apr.sh) established.
This is an allocator component port for the
[httpd/APR bug corpus](../../../bug-corpora/httpd/apr-pool-repros/README.md).
It does not execute httpd, APR's other subsystems, or any threaded use of a
pool: `APR_HAS_THREADS` and `APR_ALLOCATOR_USES_MMAP` are zero, as in the
census, and the `APR_POOL_DEBUG` implementation is not built. A
[stock CheriBSD build](host/cheribsd/README.md) exists -- the platform's own
`malloc` under every node, no adapter authority -- and there is no PoisonCap
build of APR.

## Layout and source boundary

This component uses `../../common` for verified downloads, cross toolchains,
external builds and the shared QEMU lock, in the layout the port catalog
describes. `src/native/` holds the hosted entry and the pointer adapter;
`src/capstone-domain/` the domain entry; `src/linux-guest/` the staging loader;
`src/allocators/sublet/` the authority adapter; `src/shared/` the protocol, the
metadata heap and the service stubs. Generated sources stay outside the
repository.

`upstream.json` pins the official APR 1.7.4 archive by SHA256 -- the same
archive `../fetch-apr.sh` caches, so one download serves the census and the
port. Two ordered patches, each naming its input, prerequisites and application
command:

| patch | what |
|---|---|
| `0001-freestanding-includes` | replaces the fourteen APR includes with [`../adapted/apr_shim.h`](../adapted/apr_shim.h): the census seam, recorded as a patch so the prepared source is reproducible from the archive and the patch alone. The shim is included from where the census keeps it, not copied |
| `0002-node-lifetime-hooks` | connects the node transitions to the adapter; six non-debug call sites |

## Lifetime mapping

APR's unit of storage is the **node**: `MIN_ALLOC` or more, a multiple of
`BOUNDARY_SIZE`, holding a pool's struct and everything allocated from it.
Upstream still decides everything about nodes -- the size buckets, LIFO order,
which node the next `apr_pool_create` pops. The adapter supplies the storage
under a node and, in the protected mode, the authority over it.

The hooks sit on APR's own free-list transitions, not on `free()`.
`allocator_free` files a released node on `allocator->free[index]` and
`allocator_alloc` pops it straight back; under the default
`APR_ALLOCATOR_MAX_FREE_UNLIMITED` upstream never calls `free()` on that path,
so a hook there would never fire. `aprp_node_release` runs where a node is
filed, after its header has been read and before its link is written;
`aprp_node_issue` at `have_node:`, which every node passes through on its way
to a pool. Both return the alias upstream uses from then on. `malloc`/`free` of
a node become the payload backing; of the allocator struct, the metadata heap,
so an allocator never lives in storage it hands out.

Both domain modes use the same layout and allocator code:

- `spatial`: a node keeps the alias it was carved with. A handle saved across
  `apr_pool_destroy` still names the storage, which by then is the next pool's.
- `sublet`: release and issue each `sublet_give` then `sublet_take` the node,
  so that handle is a revoked alias. Revocation clears the region; the header
  fields upstream reads across a transition, `index` and `endp`, are put back
  from the adapter's record.

`APR_ALIGN_DEFAULT` rounds to 16 here where upstream rounds to 8
(`-DAPR_ALIGN_DEFAULT_BOUNDARY=16`, a knob the shim defaults to 8 so the census
keeps upstream's numbers). A capability is 16 bytes and must be stored
16-aligned; a 24-byte `apr_palloc` followed by a `cleanup_t` -- two
function-pointer capabilities -- would otherwise trap. Native builds use the
same value so header sizes agree across the seam.

The fixed regions are 64 MiB payload, 16 MiB metadata, 8 MiB trace and 4 KiB
report; at most 8,192 nodes. `metadata` in the report is heap high-water usage.
No timing or memory-overhead claim is made.

## CheriBSD

The `cheribsd` preset builds the same allocator for CheriBSD purecap with
`src/cheribsd/node-malloc.c`: every node comes from the platform's `malloc` and
goes back through its `free`, exactly as upstream APR does. There is no payload
region and no protected mode; mode 1 is refused. This is the build that asks
the platform's own libc revocation the question at the level where it lives,
and the answer is in the corpus's
[CheriBSD results](../../../bug-corpora/httpd/apr-pool-repros/results/20260921-cheribsd/README.md):
APR never calls `free()` on the path where a destroyed pool's node is reused,
so revocation is never asked.

`bin/revocation-control` is the positive control that makes a completing stock
arm mean something. It frees a block, forces the quarantine sweep
(`malloc_revoke_quarantine_force_flush()`, the name `<stdlib.h>` deprecates
`malloc_revoke()` in favour of), and reads through the old pointer at the
corpus's own labelled `clbu`, `apr_defect_read`. With revocation on it faults
there; with it off it completes.

Two things a reader of the preset should know. `CMAKE_EXE_LINKER_FLAGS` is
`-fuse-ld=lld`, because the SDK ships `ld.lld` and no `ld`, so clang otherwise
falls back to the host's linker (`unrecognised emulation mode: elf64lriscv`);
the pymalloc PoisonCap build passes the same flag from its script. And
`allocator-example` prints `ALLOCATOR_EXAMPLE apr PASS pointer_bytes=16` as its
last line, which is what the shared runner's component registration expects.

## Build

From the repository root, source `capstone/tests/capstone-test-env.sh` and set
`CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR`, `CAPSTONE_QEMU_BINARY` and
`PORT_MUSL_ROOT` to prepared tools. From this directory:

```sh
cmake --preset native && cmake --build /tmp/capstone/apr-pools/build/native
ctest --test-dir /tmp/capstone/apr-pools/build/native
cmake --preset capstone-domain && cmake --build /tmp/capstone/apr-pools/build/capstone-domain
cmake --preset linux-guest && cmake --build /tmp/capstone/apr-pools/build/linux-guest
CHERI_SDK=... CHERI_SYSROOT=... cmake --preset cheribsd && cmake --build /tmp/capstone/apr-pools/build/cheribsd
```

Presets build under `/tmp/capstone/apr-pools/build/`. A hosted build exposes
`APR::Pools`, `bin/allocator-example` and `PORT_CLIENT_SOURCE`. The one-source
seam is `-DAPRP_CORPUS_SRC=<case.c>`: the corpus supplies `aprp_replay`, and the
build produces `bin/defects` (hosted) or `bin/defects.dom` (domain). The corpus's
`shared/build-cases.sh` invokes it once per case.

## Verification

The native suite checks patch order and duplicate rejection, that preparation
failure preserves a previous complete source, and that `allocator-example`
observes the reissue the corpus exists for: a destroyed pool's node comes back
as the next pool, `node_reuses >= 1`. Without that the example would be
measuring the wrong allocator.

The domain and CheriBSD arms are the corpus's, run and judged by
[its runners](../../../bug-corpora/httpd/apr-pool-repros/runners/capstone-domain/README.md):
`spatial` must complete, `sublet` must fault at the labelled probe, the
expected address is published by the run, and the negative control must make
every oracle fail before a pass is believed. Recorded results live with the
corpus, as summaries, never captures.
