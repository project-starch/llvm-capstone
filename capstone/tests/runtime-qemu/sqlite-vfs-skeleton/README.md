# SQLite VFS skeleton smoke

This directory is the **first Capstone-compiled SQLite-facing slice**.

It does **not** build or run the full SQLite core yet.
Instead it proves a smaller but real step:

- fetch the pinned official SQLite amalgamation through
  `capstone/benchmarks/sqlite/fetch-sqlite.sh`,
- compile a tiny custom-VFS skeleton against the real `sqlite3.h`,
- compile that code with the in-tree Capstone `clang` for `capstone64-unknown-elf`,
- and link it into a normal `.dom` sample-domain image.

## Why this exists

The current tree already validates the HostCall-oriented file/path proofs needed for a
first SQLite bridge. The next concrete question is whether the in-tree Capstone compiler
can already type-check and link a SQLite-facing VFS shim on the **domain** path.

This directory answers that question for the first narrow slice.

## Current scope

The skeleton currently validates:

- real SQLite VFS/io-method type usage from the official `sqlite3.h`,
- a runtime-initialized `sqlite3_io_methods` / `sqlite3_vfs` bundle,
- a Capstone-compiled multi-source domain build,
- and a ready-to-run QEMU wrapper for the next runtime revalidation attempt.

It intentionally does **not** yet validate:

- `sqlite3.c` compilation,
- `sqlite3_initialize()` or `sqlite3_open()`,
- HostCall-backed `xOpen` / `xRead` / `xWrite` plumbing,
- rollback-journal behavior,
- or WAL/shared-memory support.

## Important current limitation

A file-scope constant SQLite vtable initializer now compiles, but the resulting
domain still relies on tagged capability data that the current runtime path does
not preserve through domain image loading.

So this first slice still uses **runtime initialization** instead of a file-scope
constant `sqlite3_io_methods` / `sqlite3_vfs` object. That keeps the current
smoke on the validated runtime path while the deeper tagged-global-data issue is
investigated separately.

## Files

- `capstone_sqlite_vfs.h` — small SQLite-facing VFS bundle definitions.
- `capstone_sqlite_vfs.c` — runtime-initialized VFS/io-method skeleton.
- `sqlite_vfs_skeleton_domain.c` — tiny domain entry that bootstraps and self-checks the bundle.

## Build

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
bash capstone/tests/runtime-qemu/build-sqlite-vfs-skeleton.sh
```

## Run

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
bash capstone/tests/runtime-qemu/run-sqlite-vfs-skeleton.sh
```

The wrapper builds `capstone-test.user` into the shared directory before booting
QEMU, so the smoke uses the current guest loader implementation instead of any
stale copy baked into the root filesystem image.

## Next step after this skeleton

Plumb the already validated HostCall services into the skeleton callbacks:

- `xOpen` / `xClose`
- `xRead` / `xWrite`
- `xSync`
- `xTruncate`
- `xFileSize`
- `xAccess`
- `xDelete`

Once that exists, the next honest gate is a first reduced SQLite smoke, not another
purely abstract shim exercise.

