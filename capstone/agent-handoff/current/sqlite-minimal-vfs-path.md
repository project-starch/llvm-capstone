# SQLite-oriented minimal VFS path

This note turns the current "consume the validated file-service subset" guidance into
one concrete next step that is small enough to implement incrementally.

## What this means

It does **not** mean "bring up normal hosted Linux `sqlite3` first".
It means:

- compile SQLite in a reduced custom-VFS configuration,
- keep SQLite talking to a tiny domain-side VFS shim,
- have that shim use the already validated HostCall file-service operations,
- add only the smallest still-missing path semantics that SQLite actually needs,
- and defer WAL/shared-memory and broad lock choreography until a concrete run proves they are needed.

In other words, the next bridge to SQLite is:

> SQLite core -> custom Capstone VFS shim -> narrow HostCall file service

not:

> SQLite core -> full hosted libc/syscall ABI bring-up

## Why this is now the right next step

The current tree already validates:

- `FILE_OPEN`
- `FILE_READ`
- `FILE_WRITE`
- `FILE_SYNC`
- `FILE_STAT_BASIC`
- `FILE_TRUNCATE`
- `FILE_CLOSE`

That is already enough for the handle-based data path of a SQLite-style file object.
The next missing question is therefore no longer "can one more file opcode work?".
It is:

> what is the smallest SQLite-facing VFS surface that remains after those validated handle operations already exist?

The answer from a focused SQLite 3.53.1 source audit is:

- do **not** start with WAL or shared memory,
- do **not** start with a full Linux ABI mirror,
- first add the small path-level gaps SQLite still needs,
- and keep the first lock story conservative and single-actor oriented.

## Recommended first SQLite build mode

For the first bootstrap, target a deliberately narrow configuration such as:

- `SQLITE_OS_OTHER=1`
- `SQLITE_OMIT_WAL=1`
- `SQLITE_OMIT_LOAD_EXTENSION=1`
- `SQLITE_THREADSAFE=0`
- `SQLITE_TEMP_STORE=3`
- `SQLITE_OMIT_SHARED_CACHE=1`

Rationale:

- `SQLITE_OMIT_WAL=1` removes the `xShmMap` / `xShmLock` family from the first target,
- `SQLITE_TEMP_STORE=3` avoids temp-file policy work in the first pass,
- `SQLITE_THREADSAFE=0` keeps the first bootstrap about file semantics rather than mutex policy,
- `SQLITE_OS_OTHER=1` lets the project provide a custom VFS instead of pretending the hosted Linux ABI is already solved.

## Minimal VFS surface for the first Capstone shim

The first custom VFS should implement only the subset that the first rollback-journal,
single-connection SQLite run actually needs.

### File-object methods

| SQLite method | First implementation plan | Backing path |
| --- | --- | --- |
| `xOpen` | required | `FILE_OPEN` |
| `xClose` | required | `FILE_CLOSE` |
| `xRead` | required | `FILE_READ` |
| `xWrite` | required | `FILE_WRITE` |
| `xTruncate` | required | `FILE_TRUNCATE` |
| `xSync` | required | `FILE_SYNC` |
| `xFileSize` | required | `FILE_STAT_BASIC` -> `file_size` |
| `xLock` | required, but keep first version conservative | shim-local single-actor bookkeeping first, then tighten only if a real run proves it is insufficient |
| `xUnlock` | required, same scope as `xLock` | shim-local single-actor bookkeeping first |
| `xCheckReservedLock` | required, same scope as `xLock` | first return based on the same conservative single-actor lock bookkeeping |
| `xFileControl` | minimal stub | return `SQLITE_NOTFOUND` for unsupported ops; ignore harmless hints only when needed |
| `xSectorSize` | required | conservative fixed value is sufficient for first pass |
| `xDeviceCharacteristics` | required | conservative `0` first, then advertise more only if justified |

### VFS-wide methods

| SQLite VFS method | First implementation plan | Backing path |
| --- | --- | --- |
| `xOpen` | required | HostCall file-service open path |
| `xDelete` | required | **new small path service still needed** |
| `xAccess` | required at least for `SQLITE_ACCESS_EXISTS` | first focused path service now exists |
| `xFullPathname` | required | local canonical/string policy inside the shim |
| `xRandomness` | required | local deterministic or helper-fed bytes; first pass can be conservative |
| `xSleep` | required | local stub or helper call; first pass can be conservative |
| `xCurrentTimeInt64` | required | local or helper-fed time source |
| `xDlOpen`/`xDlSym`/`xDlClose` | omitted in first target | removed by `SQLITE_OMIT_LOAD_EXTENSION=1` |
| `xSetSystemCall`/`xGetSystemCall`/`xNextSystemCall` | not needed in first target | leave null in the custom VFS versioning story |

## What the next HostCall expansion should be

The next smallest runtime-facing additions are path-level, not WAL-level.
The tree now has the first focused path existence/access proof, so the next remaining
SQLite-facing path gap is:

1. delete/unlink for a path-like object

Suggested naming at the current layer:

- `runtime: add path-delete HostCall proof`

Whether those land under a `hostcall:` prefix instead is secondary; the important point is
that they are SQLite-facing runtime service additions, not generic throwaway tests.

## Concrete runtime touchpoints in the current tree

The current tree already has the exact probe shape that the next path-level proofs should copy.

Useful reference files:

- shared opcode/header definitions:
  - `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h`
- first narrow handle-metadata path:
  - `capstone/tests/runtime-qemu/hostcall-file-handle-stat-probe/hostcall_file_handle_stat_probe.smode.c`
  - `capstone/tests/runtime-qemu/hostcall-file-handle-stat-probe/hostcall_file_handle_stat_probe_guest.c`
  - `capstone/tests/runtime-qemu/build-hostcall-file-handle-stat-probe.sh`
  - `capstone/tests/runtime-qemu/run-hostcall-file-handle-stat-probe.sh`
- first narrow size-mutation path:
  - `capstone/tests/runtime-qemu/hostcall-file-handle-truncate-probe/hostcall_file_handle_truncate_probe.smode.c`
  - `capstone/tests/runtime-qemu/hostcall-file-handle-truncate-probe/hostcall_file_handle_truncate_probe_guest.c`
  - `capstone/tests/runtime-qemu/build-hostcall-file-handle-truncate-probe.sh`
  - `capstone/tests/runtime-qemu/run-hostcall-file-handle-truncate-probe.sh`
- first SQLite-facing path-access path:
  - `capstone/tests/runtime-qemu/hostcall-path-access-probe/hostcall_path_access_probe.smode.c`
  - `capstone/tests/runtime-qemu/hostcall-path-access-probe/hostcall_path_access_probe_guest.c`
  - `capstone/tests/runtime-qemu/build-hostcall-path-access-probe.sh`
  - `capstone/tests/runtime-qemu/run-hostcall-path-access-probe.sh`

The next path-level proofs should follow that same structure:

1. add new shared header constants/structs,
2. add one `.smode.c` producer for the request/response rounds,
3. add one helper-side `*_guest.c` service implementation,
4. add focused build/run wrappers,
5. validate the observable effect, not just the opcode dispatch.

## Why lock choreography is still deferred

SQLite does still call:

- `xLock`
- `xUnlock`
- `xCheckReservedLock`

in rollback-journal mode.

However, that does **not** automatically mean the next ABI should freeze a general multi-actor
locking protocol.

The first honest bootstrap target is narrower:

- one connection,
- one domain-side SQLite instance,
- no WAL,
- no shared-memory coordination,
- no claim yet of multi-process correctness.

For that target, a conservative single-actor lock model is the smallest practical step.
Only if a real SQLite smoke run proves that this is too weak should the branch add
protocol-level lock acquire/upgrade/release semantics.

## Recommended implementation order

### Step 1: add the first path services

Add and validate the next narrow path operation for:

- delete/unlink.

These should get the same focused runtime/QEMU treatment as the existing file-handle proofs.

### Step 2: add a tiny SQLite VFS shim

Create a small domain-side shim that:

- stores the helper-managed HostCall handle inside the VFS file object,
- maps `xRead`/`xWrite`/`xSync`/`xTruncate`/`xFileSize`/`xClose` onto the validated HostCalls,
- provides local `xFullPathname`,
- and implements conservative first-pass lock bookkeeping.

### Step 3: run one honest SQLite smoke scenario

The first smoke scenario should stay deliberately narrow:

- open/create a database,
- create one table,
- insert one row,
- close the database,
- reopen it,
- read the row back,
- verify that rollback-journal cleanup still behaves correctly.

That scenario is a much better next gate than adding more speculative opcodes in isolation.

## What should not come first

Do **not** jump first to:

- WAL shared-memory support,
- `xShmMap` / `xShmLock`,
- a full hosted Capstone Linux libc/syscall ABI bring-up,
- broad multi-actor lock choreography,
- or a large POSIX mirror in the HostCall ABI.

Those may become necessary later, but they are not the smallest next bridge from the
already validated file-service subset to a real SQLite consumer.

## Short version

"SQLite-oriented minimal VFS path" means:

- keep SQLite in a reduced custom-VFS build,
- reuse the current handle-based HostCall file service,
- keep the now-validated path existence/access proof,
- add only path delete/unlink next,
- then prove a real SQLite smoke scenario,
- and only after that decide whether stronger lock semantics are truly needed.



