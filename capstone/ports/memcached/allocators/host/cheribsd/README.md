# memcached allocators on CheriBSD

Build a direct-link library and [standalone example](../../examples/allocators.c)
using the [shared CheriBSD workflow](../../../../common/host/cheribsd/README.md).
From this component directory, after sourcing the repository test environment:

```sh
export CHERI_SDK=/path/to/sdk
export CHERI_SYSROOT=/path/to/rootfs-riscv64-purecap
bash host/cheribsd/build.sh /tmp/capstone/cheribsd/memcached
bash host/cheribsd/run.sh /tmp/capstone/cheribsd/memcached /tmp/capstone/memcached-cheribsd-run-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image /path/to/cheribsd.img
```

The CMake library target is `Memcached::Allocators`; `--client
/absolute/path/main.c` links your own `bin/allocator-client` through it.
`cmake --preset cheribsd` is an equivalent configure entry point.

**This is the stock build, and that is its point.** Slab pages and cache
objects come from the platform's own `malloc` and go back through its own
`free`, exactly as upstream memcached does (`src/cheribsd/malloc-leases.c`);
chunks are offsets into a malloc'd page, addressed through the page's
capability, as on any CheriBSD build of memcached. There is no payload region
and no adapter authority. libc revocation is therefore asked the question at
the level where it lives — and neither allocator calls `free()` on the path
where a freed unit is reused: slabs pushes the chunk on its class's list,
cache.c pushes the object on its `STAILQ`. Mode 1 is refused. There is no
PoisonCap build of these allocators.

`bin/revocation-control` is the positive control that makes a completing stock
arm mean something: it frees a block, sweeps, and reads through the old pointer
at the corpus's own labelled `clbu`. With `--runtime-revocation on` it must
fault there; with `off` it must complete. A stock arm that completes beside a
control that faults says "the mechanism is active and did not fire", which is a
different sentence from "there is no mechanism".

The SDK ships `ld.lld` and no `ld`; the port adds `-fuse-ld=lld` itself for
this platform, so the shared `build.py` — which configures from the toolchain
file, not the preset — links with the SDK's linker too.

The [memcached corpus](../../../../../bug-corpora/memcached/allocator-repros/runners/cheribsd/README.md)
runs its cases through this build.
