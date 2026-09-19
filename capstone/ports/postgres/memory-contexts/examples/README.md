# Allocator client examples

These are ordinary PostgreSQL allocator clients, not trace replay programs.
Each client is compiled unchanged for native execution, a spatial Capstone
domain and a Sublet Capstone domain. The examples perform only valid accesses;
every build should finish with `PG_CLIENT <name> RESULT 0`.

Start with the client files; `support/` contains the platform setup.

| Client | Demonstrates | Lifetime rule |
|---|---|---|
| [allocset.c](allocset.c) | Growable buffers, `repalloc`, `palloc`, context switching | Individual free, or bulk reset/delete |
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
or change behavior based on the build mode. `support/domain.c` receives the
shared arena and chooses spatial backing or Sublet initialization. Sublet must
consume the linear arena directly in the region-share handler. Keep that code
in the wrapper when writing a new client.

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

The five selected CTests are the four client programs and the runner's
positive/negative verdict controls. The full native suite remains available
without the label filter. See the [component README](../README.md) for build
prerequisites and the pinned upstream download.

## Capstone/Sublet under QEMU

Set `CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR` and
`CAPSTONE_QEMU_BINARY` to the configured Capstone tools. Activate a Python
3.11.4+ environment with `pexpect` installed before these commands:

```sh
source capstone/tests/capstone-test-env.sh
python3 -c 'import pexpect'
cmake --preset capstone-domain -S capstone/ports/postgres/memory-contexts \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build /tmp/capstone/postgres-memory-contexts/build/capstone-domain
cmake --preset linux-guest -S capstone/ports/postgres/memory-contexts
cmake --build /tmp/capstone/postgres-memory-contexts/build/linux-guest
ctest --test-dir /tmp/capstone/postgres-memory-contexts/build/capstone-domain \
  -L client-examples --output-on-failure
```

This runs four CTests, each with spatial and Sublet execution: eight QEMU arms.
To run just the Slab client with Sublet:

```sh
python3 capstone/ports/postgres/memory-contexts/examples/run-qemu.py slab \
  --mode sublet \
  --domain-build /tmp/capstone/postgres-memory-contexts/build/capstone-domain \
  --linux-build /tmp/capstone/postgres-memory-contexts/build/linux-guest \
  --output /tmp/capstone/postgres-client-results
```

Names are `allocset`, `generation`, `slab`, and `bump`; `--mode` defaults to
`both`. The runner serializes QEMU with the shared lock and retains staged
binaries, fingerprints, serial output and verdicts per attempt. A pass requires
the correct client's successful result, no capability fault, and completion
of host cleanup. Failed attempts are not automatically retried.

The existing guest loader requires an input-file argument, so the runner
supplies an empty, unused input file. It is not an A11 trace. These programs
do not need the replay engine or its identity tables.

For nondefault build directories, pass matching `--domain-build` and
`--linux-build`; configure the domain build with `-DPG_LINUX_BUILD_DIR=...` so
CTest finds the loader. Both builds must use identical region settings.

## Write another client

1. Add a C file implementing the `client_run` interface in `client.h`.
2. Add its name to the list in `cmake/Examples.cmake` and the runner's client
   choices. The existing wrappers and build wiring supply all three versions.
3. Add the C file to `ledger.manifest` as example/test code.
4. Build and run the native and paired QEMU tests above.

Use the separate `security-tests/` fixtures for deliberate stale-pointer
accesses: those need an exact-access fault verdict, not the successful-return
verdict used here. These examples are freestanding allocator clients, not a
complete PostgreSQL server or arbitrary hosted Linux applications. The current
ports require PostgreSQL release-layout headers and have fixed metadata limits.

## Verified on 2026-09-18

All four clients passed natively and in both QEMU modes (eight QEMU arms,
without retries). The full native suite passed 13/13 CTests, including the
runner's negative controls. Native uses the Release preset; domain validation
uses the Debug preset with PostgreSQL release-layout headers. No silicon or
performance result is implied.
