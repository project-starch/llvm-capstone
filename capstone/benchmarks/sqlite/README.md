# SQLite in-memory smoke

## Status

SQLite 3.53.3 now **compiles and links** for `capstone64-unknown-elf`. The
QEMU smoke is not yet green: initialization reaches
`sqlite3RegisterBuiltinFunctions()`, then faults while copying an aggregate
whose nested capability fields were not tagged.

This is a compiler/runtime coverage blocker, not a filesystem or VFS blocker.
No file-backed database path is used.

## Workload

The domain is intended to run:

1. `sqlite3_open(":memory:")`;
2. `CREATE TABLE items(name TEXT, value INTEGER)`;
3. three `INSERT`s;
4. `SELECT name,value FROM items`;
5. row validation, `sqlite3_finalize()`, and `sqlite3_close()`.

The host wrapper prints rows through two shared regions. A successful future
run must print:

```text
row name=alpha value=11
row name=beta value=22
row name=gamma value=33
__CAPSTONE_SQLITE_MEMORY_PASSED__
```

## Source and build

`fetch-sqlite.sh` downloads the official
`sqlite-amalgamation-3530300.zip`, verifies SHA3-256
`d45c688a8cb23f68611a894a756a12d7eb6ab6e9e2468ca70adbeab3808b5ab9`,
and extracts it under `$CAPSTONE_TMP_ROOT/sqlite-src`. SQLite is not vendored.
The existing runtime-QEMU SQLite VFS skeleton uses the same fetcher.

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/benchmarks/sqlite/build-sqlite-capstone.sh
bash capstone/benchmarks/sqlite/build-sqlite-host.sh
```

The target build uses `SQLITE_OS_OTHER=1`, `SQLITE_THREADSAFE=0`,
`SQLITE_DEFAULT_MEMSTATUS=0`, `SQLITE_TEMP_STORE=3`,
`SQLITE_ZERO_MALLOC=1`, and `SQLITE_ENABLE_MEMSYS5=1`. It omits load
extensions, localtime, mmap, WAL, shared cache, temp DBs, foreign keys, JSON,
UTF-16, incremental blobs, `sqlite3_get_table()`, deprecated APIs, and
`EXPLAIN`.

SQLite memsys5 manages a 1 MiB aligned arena selected with
`SQLITE_CONFIG_HEAP`. Common memory/string routines come from the existing
BEEBS `beebs_freestanding_string.c`; the local shim adds only the remaining
ctype/string/time declarations SQLite requires. The build reuses the existing
runtime-initialized SQLite VFS skeleton. Its `xOpen` rejects files, which is
valid for `:memory:`.

`SQLITE_OMIT_FLOATING_POINT` still leaves floating literals that lower to
software libcalls. The build links the narrow required set from the in-tree
compiler-rt builtins.

## Generated-source adaptations

The downloaded amalgamation remains unchanged. The build creates
`$CAPSTONE_TMP_ROOT/sqlite-build/sqlite3-capstone.c` with bounded adaptations:

- use a real identity-only function for `SQLITE_TRANSIENT` instead of the
  capability function-pointer value `-1`;
- correct SQLite 3.53.3's `z`/`zIn` typo in its no-floating-point path;
- replace capability-valued pointer ternaries and one boolean pointer
  subtraction with equivalent control flow;
- disable Lemon's dynamic parser stack and use `YYSTACKDEPTH=1000`;
- use capability cursors for two sorter-local same-array index differences;
- gate atomic detection on `SQLITE_THREADSAFE`;
- move integer-valued built-in `pUserData` fields out of static initializers;
- initialize memsys5's function-pointer methods table at runtime.

The last two avoid unsupported 128-bit integer-pointer constants and one
non-recursive capability-global shape. They do not fix the general blocker
below.

## Runtime blocker

Run the current smoke with:

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/benchmarks/sqlite/run-sqlite-memory.sh
```

Current result: QEMU aborts before any SQL statement executes. The progression
is:

1. A static `sqlite3_mem_methods` struct initially produced
   `cs.cjalr requires capability` in `sqlite3MallocInit()` at
   `sqlite3GlobalConfig.m.xInit(...)`.
2. Runtime-initializing that table advances into built-in registration.
3. SQLite's built-in table is an array of structs containing function and
   string pointers. `CapstoneCapGlobalInit` does not recursively materialize
   nested aggregate fields. Clang also lowers a local aggregate copy to
   `memcpy` from a private global template containing the same untagged
   pointers.
4. The current terminal failure is `helper_cscincoffset: Assertion
   'rs1_v->tag' failed` in `memcpy`, while
   `sqlite3RegisterBuiltinFunctions()` copies that template.

`probes/nested-cap-global.c` is the 592-byte minimal reproducer. It defines an
array of structs with one function pointer and one string pointer. It compiles
and links, then QEMU fails with:

```text
[CAPSTONE] cs.cjalr requires capability in rs1
```

The required compiler follow-up is recursive materialization of capability
fields inside arbitrary global aggregates, including compiler-generated private
constant templates. Until that exists, adding more SQLite-specific table
rewrites would hide the general compiler gap.
