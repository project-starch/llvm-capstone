# PostgreSQL memory-context component

PostgreSQL's memory managers (`src/backend/utils/mmgr`: AllocSet, Generation, Slab, Bump and the
`mcxt.c` dispatch) built outside the server, with a replay of recorded workloads, client
examples and the build seam the [mmgr defect corpus](../../../bug-corpora/postgres/mmgr-repros)
uses. The [server port](../app) applies the same Sublet patch to 17.5.

The component and its native defect corpus use **PostgreSQL 17.0**, pinned by URL and SHA-256 in
`upstream.json`. The shell scripts in `../` read the same file through `../upstream.sh`; a
conflicting `PG_VERSION` is rejected. 17.0 is intentional: it retains consumer lifetime defects
fixed in later 17.x releases (see `../../../docs/ref/postgres-nested-allocator-defects.md`). Its
manager sources are identical to 17.5's. `port-origin.json` describes the original **17.5**
adaptation.

## Targets

| preset | what it builds |
|---|---|
| `capstone-application` | the replay, the clients and the corpus cases as Capstone processes on the virtual profile (`CAPSTONE_SDK`); `-DPG_SUBLET=ON` adds the Sublet protection |
| `cheribsd` | the same for stock CheriBSD purecap ([host/cheribsd](host/cheribsd/README.md)) |
| `native` | the unpatched managers, the replay, the clients and the tests |

Every target builds the `PostgreSQL::MemoryContexts` library and `allocator-example`
(`examples/contexts.c`), the link example the shared CheriBSD harness runs; a custom main can be
linked through `PORT_CLIENT_SOURCE`.

## Patches

`cmake/prepare-source.py` verifies the archive and the unmodified manager sources, then builds
each variant as a mirror of the manager files with its ordered patches applied:

| patch | what it does | `spatial` | `sublet` |
|---|---|:-:|:-:|
| 0001 allocset-capstone-size-classes | the smallest AllocSet chunk holds a 16-byte free-list link | x | x |
| 0002 memorychunk-capstone-alignment | the chunk header in the capability layout | x | x |
| 0003 memory-contexts-sublet-lifetimes | the Sublet protection: every chunk a child lifetime of its block (`CDERIVE`), revoked by `pfree` and `repalloc` (`CREVOKE`) | | x |

The capability platforms build `spatial`, or `sublet` with `PG_SUBLET=ON` (capstone-application
only); native builds the pristine managers. Patch 0003 works at the dispatch: `mcxt.c` issues a
child bounded to the request on the way out of every allocation entry and revokes it on the way
back in, handing the context method the allocator's own pointer at the same address. The context
types only report their blocks to `sublet.c` (add, forget before `free`, renew on a reset that
keeps the block), because `pfree` has nothing but the pointer and a bounded pointer cannot reach
the chunk header in front of it. Its header states the reasons in full.

The patches require PostgreSQL's release-layout configuration; `MEMORY_CONTEXT_CHECKING` and
`CLOBBER_FREED_MEMORY` are not supported.

## Build

From the repository root, after sourcing `capstone/tests/capstone-test-env.sh`. Native builds need
CMake 3.25+, Ninja, Python 3.11.4+, a C compiler, make, patch and PostgreSQL's configure
prerequisites (including bison and flex):

```sh
cmake --preset native -S capstone/ports/postgres/memory-contexts
cmake --build /tmp/capstone/postgres-memory-contexts/build/native
ctest --test-dir /tmp/capstone/postgres-memory-contexts/build/native --output-on-failure
cmake --preset capstone-application -S capstone/ports/postgres/memory-contexts \
  -DCAPSTONE_SDK=<virtual SDK> [-DPG_SUBLET=ON] [-DPG_CORPUS_DIR=<corpus>]
cmake --build /tmp/capstone/postgres-memory-contexts/build/capstone-application
```

The native suite checks the pin and the source-integrity guards, replays the fixture traces
against their recorded counts and payloads, and runs the four [client examples](examples/README.md).
The protected variant is exercised on the virtual platform by the mmgr corpus's
`virtual-pg-pools` arm.

## History

Until 2026-10-11 this component also carried a lifetime adapter (patches 0003-0007 over the
context pools in `src/allocators/sublet/`), a freestanding Capstone domain target with its
security tests, memory profiler and linux-guest loader, and a CheriBSD PoisonCap backend.
`CDERIVE`/`CREVOKE` made the adapter unnecessary, and the other targets are not part of the
virtual platform. Their recorded results stay in `results/`; the 2026-09-18 memory-profile
campaign measured **17.5** and keeps that label.
