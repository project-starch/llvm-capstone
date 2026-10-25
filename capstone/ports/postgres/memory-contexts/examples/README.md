# Allocator client examples

These are ordinary PostgreSQL allocator clients, not trace replay programs.
Each client is compiled unchanged for every target of the component, with or
without the Sublet patch. The examples perform only valid accesses; every
build should finish with `PG_CLIENT <name> RESULT 0`.

Start with the client files; `support/` contains the platform setup.

| Client | Demonstrates | Lifetime rule |
|---|---|---|
| [allocset.c](allocset.c) | Growable buffers, `repalloc`, `palloc`, context switching | Individual free, or pool reset/delete |
| [generation.c](generation.c) | FIFO-like batches of messages | Individual free; whole blocks become recyclable when empty |
| [slab.c](slab.c) | Replacing jobs in a fixed-size table | Every allocation requests the configured object size |
| [bump.c](bump.c) | Per-request scratch and a surviving scalar summary | Reset/delete only; no `pfree` or `repalloc` |

## The client boundary

Every file defines `int client_run(MemoryContext parent)`. The caller provides
a live parent, initializes `TopMemoryContext`/`CurrentMemoryContext`, and checks
the return value. The client creates and deletes its own child contexts without
deleting the parent's context. In a PostgreSQL backend the surrounding backend
already supplies that setup; these standalone executables use the wrappers.

`MemoryContextAlloc(context, bytes)` makes ownership explicit. `palloc(bytes)`
instead uses `CurrentMemoryContext`; the AllocSet example saves and restores it
with `MemoryContextSwitchTo`. Restore it before deleting the selected context.
After a moving `repalloc`, only its returned pointer should be retained. After
free/reset/delete, all old aliases are invalid; assigning a local pointer to
NULL is good client hygiene, but Sublet—not that assignment—revokes the aliases.

The client files do not include Sublet headers, perform capability operations,
or change behavior based on the build mode: with patch 0003 the managers
protect their chunks themselves, and the client's code is the same.

## Native build and execution

From the repository root, on the branch containing these examples:

```sh
source capstone/tests/capstone-test-env.sh
cmake --preset native -S capstone/ports/postgres/memory-contexts
cmake --build /tmp/capstone/postgres-memory-contexts/build/native
ctest --test-dir /tmp/capstone/postgres-memory-contexts/build/native \
  -L client-examples --output-on-failure

# Or run a single program:
/tmp/capstone/postgres-memory-contexts/build/native/bin/client-slab-native
```

The four selected CTests are the four client programs. The full native suite
remains available without the label filter. See the [component README](../README.md)
for build prerequisites, the pinned upstream download and the capability targets.

## Write another client

1. Add a C file implementing the `client_run` interface in `client.h`.
2. Add its name to the list in `cmake/Examples.cmake`. The existing wrapper
   and build wiring supply every target.
3. Add the C file to `ledger.manifest` as example/test code.
4. Build and run the native tests above.

Deliberate stale-pointer accesses belong in the
[mmgr defect corpus](../../../../bug-corpora/postgres/mmgr-repros): those need an
exact-access fault verdict, not the successful-return verdict used here.
