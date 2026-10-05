# SQLite 3.22.0 as an ordinary program

SQLite 3.22.0 is the release of the published PoisonCap artifact and the anchor of the memory
campaigns. This component builds it the way it is built on any Linux system, and runs it on the
delegated runtime as a Linux command:
- the `sqlite3` shell and `speedtest1`;
- SQLite's own defaults, the unix VFS and musl;
- database files, POSIX record locks, temporary files and floating point.

The other SQLite builds in [`..`](../README.md) are freestanding domains, and nothing of them is
used here.

| | freestanding builds (`../build-sqlite-*.sh`) | this build |
|---|---|---|
| OS layer | `SQLITE_OS_OTHER`, a VFS that refuses files | the unix VFS: files, locks, journals, temporary files |
| C library | a local shim and BEEBS string routines | musl, through the application SDK |
| allocator | memsys5 on a static arena | `malloc`, the runtime's heap (the first-fit heap, `level0.c`, by default) |
| floating point | omitted (`SQLITE_OMIT_FLOATING_POINT`) | on |
| threads | `SQLITE_THREADSAFE=0` | SQLite's default (1) |
| temporary data | in memory (`SQLITE_TEMP_STORE=3`) | SQLite's default (files) |

The SDK's `capstone-cc` compiles every application with `-ffreestanding -fno-builtin`, so that
clang assumes no C library it cannot see. That is a compiler convention. The program is hosted:
it has `main`, musl's headers and library, and the unix OS layer.

## What it needs from the platform

This branch is stacked on two changes:
- `musl-fcntl-lock-pointer`: musl patch 0005. `fcntl` passed the `struct flock` pointer as a
  number, and SQLite locks through it in every transaction.
- `runtime-malloc-usable-size`: the first-fit heap's `malloc_usable_size`.

Both are on `delegation-v0-removal-stack` (#144), whose delegated rows serve every system call
the unix VFS makes. The threads stack alone lacks `fchmod`, `fchown` and `fallocate`.

## Source and configuration

`prepare-sources.sh` fetches both archives, pinned by SHA3-256:
- the amalgamation `sqlite-amalgamation-3220000.zip`;
- `test/speedtest1.c`, from the full release `sqlite-src-3220000.zip`, with the result oracle
  `../study/speedtest1-oracle.patch`.

It then applies these source changes:
- [`../adapt-sqlite-322.sh`](../adapt-sqlite-322.sh), the Capstone adaptations every 3.22.0
  build carries: `saveBuf` alignment, `BtCursor` alignment after the `VdbeCursor`, the `RowSet`
  size, and the `sqlite3_filename` typedef.
- **Sorter records at 16-byte offsets.** With temporary data in files, the sorter packs its
  records into one buffer at `ROUND8` offsets, and every record holds a pointer 16 bytes in. A
  record 8 bytes off its boundary faults at its first link: a misaligned store, cause 6, in
  `vdbeSorterSort`, reached by the first `CREATE INDEX`. The freestanding builds define
  `SQLITE_TEMP_STORE=3` and never take this path. The rewrite is not in `adapt-sqlite-322.sh`,
  because the recorded freestanding images are built from that script's output.

Configuration: `HAVE_MALLOC_H` and `HAVE_MALLOC_USABLE_SIZE`, which configure defines on Linux.
Without them SQLite puts an 8-byte size header in front of every block. Every structure it
allocates that holds a capability then starts 8 bytes off its 16-byte boundary, and the shell
faults at its first mutex (cause 6, `pthread_mutex_init`). The native oracle uses the same two
defines.

## Build and run

```sh
source capstone/tests/capstone-test-env.sh
bash capstone/ports/sqlite/app/prepare-sources.sh "$CAPSTONE_TMP_ROOT/sqlite-322"
bash capstone/ports/sqlite/app/build-native.sh "$CAPSTONE_TMP_ROOT/sqlite-322" "$CAPSTONE_TMP_ROOT/sqlite-322-native"
SQLITE_OPT_LEVEL=-O2 bash capstone/ports/sqlite/app/build-domain.sh <sdk> \
  "$CAPSTONE_TMP_ROOT/sqlite-322" "$CAPSTONE_TMP_ROOT/sqlite-322-app"
python3 capstone/ports/sqlite/app/run-app.py --state "$VM_STATE" --share <the VM's share> \
  --build "$CAPSTONE_TMP_ROOT/sqlite-322-app" --native "$CAPSTONE_TMP_ROOT/sqlite-322-native" \
  --size 1 --report app.json
```

`<sdk>` is `ports/common/application/build-sdk.sh`'s output. `run-app.py` compares each check with
the native programs' output, byte for byte:
- **`work`**: `work.sql.in` on a database file. It covers types, floating point, transactions, an
  index, a recursive CTE, string functions, update and delete, `integrity_check`, `.tables` and
  `.schema`. While the shell holds `BEGIN IMMEDIATE`, it starts a second writer through
  `.system` (`second-writer.sh`), and that writer must be refused ("database is locked").
- **`again`**: a second shell on the same file.
- **`speedtest1`**: the main testset on a database file. For each of its 32 phases the oracle
  records the row count and an FNV-1a hash of every result cell.

## Results

Recorded in [`results/20260930-sqlite-322-app.json`](results/20260930-sqlite-322-app.json).
- At `-O2` and `-O0`, all three checks match the native oracle.
- Each change above has a failing control in the same record:
  - without the configuration, the first mutex faults;
  - without the sorter rewrite, the first `CREATE INDEX` faults;
  - linked against musl without patch 0005, the first lock faults.

## Not covered

- **WAL.** It maps the `-shm` file `MAP_SHARED`, and a domain maps no files. Only
  `locking_mode=EXCLUSIVE`, which keeps the WAL index in heap memory, could work, and it is
  untested.
- **Memory-mapped I/O** (`mmap_size` above 0) is untested for the same reason.
- **`load_extension`.** There is no dynamic loading.
- **More than one thread.** This base has none.
- **Not run yet:**
  - the sqllogictest corpus with this build;
  - `speedtest1` above size 1, and testsets other than `main`;
  - the Sublet heap, which has no `malloc_usable_size` yet.
