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
bash capstone/ports/sqlite/build-sqlite-capstone.sh
bash capstone/ports/sqlite/build-sqlite-host.sh
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
bash capstone/ports/sqlite/run-sqlite-memory.sh
```

Current result: **green** — the domain runs the base 3-row workload and the
extended workload, emitting `__CAPSTONE_SQLITE_EXTENDED_PASSED__` then
`__CAPSTONE_SQLITE_MEMORY_PASSED__`.

## speedtest1 and the Sublet port

`run-sqlite-speedtest1.sh` runs SQLite's own benchmark in a domain, `--memdb --size 1`,
all 32 tests, with the lookaside pool on (`SQLITE_LOOKASIDE=1200,40`) and memsys5 as the
allocator under it, and cuts SQLite's own statistics from the payload. `speedtest1_domain.c`
includes `test/speedtest1.c` from the SQLite source tree unchanged under another `main`, with
stdio on the hostcall payload, fixed arguments, and the double-to-int64 mapping `sqlite3.h`
applies without floating point. Three instruments stay: the host's `--tail` mode prints the
payload while the domain runs, with a heartbeat; `exit()` returns to the host from anywhere,
so a fatal error's message is read instead of parking the domain in `abort()`;
`-DSPEEDTEST1_STOP_AT=n` returns at milestone n.

Two seams are the runner's, what goes in them is an experiment's: `SPEEDTEST1_HOOK=1` with
`SPEEDTEST1_HOOK_SRC` and `SQLITE_HOOK_PATCH` links an instrument in beside the domain (it
defines `speedtest1_hook_table_bytes`, `speedtest1_hook_install` and
`speedtest1_hook_report`, and gets a table carved beside the arena) and patches its calls
into copies of the sources; `SPEEDTEST1_PROBE=n` with `SPEEDTEST1_PROBE_SRC` hands the run to
`speedtest1_probe` instead of the benchmark. The domain exports its two payload writers for
whatever is linked in. A1's instrument and its six probes are
`experiments/a1-sqlite-reuse/`, whose `run.sh` sets all of it.

`SPEEDTEST1_SUBLET=1` runs the same benchmark on the Sublet port of both allocators. Two
different ports live in this directory and are kept apart on purpose: everything else here
makes SQLite compile and run on Capstone, unprotected; `sublet/` makes it protected, applied on
top of that and counted separately, because the paper's A7 counts exactly those lines.

| File | What it is |
|---|---|
| `sublet/` | the Sublet port: `sublet.h`, the primitives as operations on capability slots (split, take, give, handle, carve, move; no linear capability ever sits in a C variable), and `sublet-3530300.patch`, 28 hunks against the sed-adapted amalgamation, each classed in its header. Its README carries the recipe and the bookkeeping |
| `speedtest1_domain.c` | the adapter: freestanding stdio, the exit that returns, the arena; takes memsys5's heap or the port's pool from the host's region (`sqlite3_sublet_grant`) and memsys5's tables and an instrument's table from a second, carves them from the stack region when the host lends none, prints the primitive counts |
| `sqlite_host.c --tail [--pool <bytes> \| --arena <bytes>] [--tables <bytes>]` | creates the regions and shares them with a handle the monitor keeps, `--pool` non-linear (REV_DEFAULT), `--arena` linear (REV_BORROWED), and releases them after the run |

Under the port memsys5's pool is one linear capability carved into blocks, one revocation
node each; a split takes a handle senior to both halves first, so a merge is one revoke;
every hand-out is mrev and delin, every free a revoke. The lookaside's block comes from
memsys5 linear (`sqlite3MallocLinear`), is carved into slots the same way, and dies with
one revoke when memsys5 frees it. The allocators' policies are unchanged: an instrument's
per-test profile under the port is the unprotected build's to the allocation, plus one
memsys5 allocation per connection for the lookaside's side table.

Two things the emulator hides, found in review: its revoke leaves an uninitialised
region's cursor at the end so one `init` reclaims it, where the specification wants the
region written through first (a merge would cost a write of the block on such hardware);
and its stores of linear capabilities do not null the source register, so a helper that
reads a register after storing it works here and not there (`sublet_take_linear` reads
first). The first has since arrived in the emulator: the merge line's Q-07 (capstone-qemu
72fb56be86) moves the cursor to the base on a revoke and makes `init` trap, and under it the
port halts at its first merge with cause 29 until it writes the block through before `init`.
The passes ran on the pre-Q-07 diagnostic build. The emulator's node pool is sized at start
(`CAPSTONE_REV_NODES`, the runner sets eight million) and reuses no node; a run at `--size 1`
takes 43417 nodes, at `--size 100` 3525357.

The allocators' memory comes from the host as regions, so the benchmark's size is not
bounded by the domain's stack region. `SPEEDTEST1_POOL` is memsys5's heap in the unprotected
build (`--pool`, non-linear, control bytes inside, POOL/65 atoms of 64 bytes); the port's
pool is the same atoms times 64 (`--arena`, linear under a handle the monitor keeps); a
tables region (`--tables`) holds memsys5's tables under the port and what sits beside the
pool. Above 4 MiB a region comes from the kernel's CMA area (`cma=1G` on
the guest's command line, which the runner passes), and after the run the host gives both
back with `release_region`. For `--size 100` set `SPEEDTEST1_POOL=136314880` (2^21 atoms,
130 MiB) and `SPEEDTEST1_ARGS` accordingly; the run takes about seven minutes on QEMU with
the tag map indexed (capstone-qemu branch `diag/domain-runs`), over an hour without. Without
host regions, `-DSPEEDTEST1_STACK_ARENA`
carves 2 MiB from the stack region's low end, which fits `--size 1`.

The plain benchmark at `--size 1` runs on the QEMU `dev` pins. A run with an instrument
linked in has needed the diagnostic QEMU's `CAPSTONE_GP_NONLIN=1` (capstone-qemu branch
`diag/domain-runs`): with the default gp fabrication every code capability is linear after
the first return to the entry frame, and the compiler's `movc` copy of a live one then nulls
its source, which halted the domain in `sqlite3RunVacuum`.

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
