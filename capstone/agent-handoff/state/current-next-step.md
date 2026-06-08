# Current recommended next step

## Milestone

Build a minimal SQLite-oriented VFS shim that consumes the already-validated
file-service subset (`OPEN`, `READ`, `WRITE`, `SYNC`, `STAT_BASIC`, `TRUNCATE`, `CLOSE`
plus `PATH_ACCESS` and `PATH_DELETE`) through the existing HostCall boundary.

## Why this is next

All individual file-service operations are individually validated. The next unresolved
question is not "can another toy proof work?" but "does the composition hold for a real
consumer?" A tiny SQLite VFS shim is the smallest meaningful real consumer.

## Concrete first actions

1. Create `capstone/capstone-c/sqlite_vfs/` (or equivalent location).
2. Implement `xOpen`, `xRead`, `xWrite`, `xSync`, `xFileSize`, `xTruncate`, `xClose`
   using the existing HostCall opcodes — no new ABI surface.
3. Implement `xAccess` and `xDelete` using the existing `PATH_ACCESS` / `PATH_DELETE` ops.
4. Keep `xFullPathname` as a local shim (no cross-boundary call needed).
5. Defer `LOCK_ACQUIRE` / `LOCK_RELEASE` until a concrete consumer proves they are required.

## Exit criterion

- The shim compiles and links against the validated HostCall boundary.
- A smoke test (open a file, write, sync, read back, close) passes inside QEMU.
- No new HostCall opcodes were added; all used opcodes are already validated.

## Design detail

Full design: `design/sqlite-minimal-vfs-path.md`
Wire spec: `design/hostcall-file-service-v0-wire-spec.md`

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` — its absence silently
switches the image to stock OpenSBI and breaks all runtime proofs.
