# SQLite in-memory smoke

## Status

SQLite 3.53.3 **compiles, links, and runs end to end** for `capstone64-unknown-elf`
as a pure-capability domain. The QEMU smoke is **green**: `run-sqlite-memory.sh`
emits both `__CAPSTONE_SQLITE_EXTENDED_PASSED__` and
`__CAPSTONE_SQLITE_MEMORY_PASSED__`, returning correct rows. All 8 capability
cap-tag / alignment gaps that once blocked init (the last being the aggregate copy in
`sqlite3RegisterBuiltinFunctions`) are resolved — see `state/current-state.md`
(§"SQLite in-memory bring-up") and the dated `history/` gap notes (gaps 1–9).

The extended in-domain workload exercises transactions, a secondary `INDEX`,
`INTEGER PRIMARY KEY`+`REAL`, bound prepared inserts, `UPDATE`/`DELETE`, aggregates +
sorter, index-driven `WHERE`, `JOIN`, `GROUP BY`, and string functions. No
file-backed database path is used (in-memory only).

Residual notes: bindings must use `SQLITE_STATIC`, not `SQLITE_TRANSIENT` (gap 9,
a `.h`/`.c` sentinel mismatch in the TRANSIENT patch); the 8-byte-alignment class
(gaps 6/8) may surface more instances under wider workloads (e.g. `speedtest1`).

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

The last two avoided unsupported 128-bit integer-pointer constants and one
non-recursive capability-global shape.

## Run

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/benchmarks/sqlite/run-sqlite-memory.sh
```

Current result: **green** — the domain runs the base 3-row workload and the
extended workload, emitting `__CAPSTONE_SQLITE_EXTENDED_PASSED__` then
`__CAPSTONE_SQLITE_MEMORY_PASSED__`.

### Resolved blocker history (kept for reference)

Init once aborted before any SQL executed, at the built-in-function registration
copy: `helper_cscincoffset: Assertion 'rs1_v->tag' failed` inside a `memcpy`,
while `sqlite3RegisterBuiltinFunctions()` copied a private aggregate template
whose nested capability fields were untagged. The general fixes landed as gaps
1–2 (`CapstoneCapGlobalInit` recurses nested global aggregates; clang's
memcpy-from-private-template of cap aggregates is handled) plus the QEMU
tag-preserving `ldc`/`stc` (gaps 3–4) and the alignment fixes (gaps 6/8). The
minimal reproducer `probes/nested-cap-global.c` (an array of structs with a
function pointer and a string pointer) now runs without fault. Full per-gap
trail: `state/current-state.md` and the dated `history/` gap notes.
