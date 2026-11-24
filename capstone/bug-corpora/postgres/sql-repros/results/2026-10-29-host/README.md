# 2026-10-29: mmgr-repros' five live defects, asked to run whole

`../../../mmgr-repros` holds five defects live at 17.5 as C programs against the real memory
contexts. This run asked of each: does upstream's own trigger reach it in the whole backend, as the
Capstone port runs it -- `postgres --single`, one process, no postmaster? Two do: case 10, and case 11
through a sized transaction instead of upstream's recompiled threshold (last section).

## Builds and runs

PostgreSQL 17.5 from the port's tarball (sha256 `fcb7ab38e23b264d1902cb25e6adafb4525a6ebcbd015434aeef9eda80f528d8`)
on p13, gcc 15.2, `CFLAGS='-O1 -g'`, `--without-icu --without-readline`:

- with `--enable-cassert --enable-debug`;
- without assertions, as the port is built, with `--enable-debug CPPFLAGS=-DUSE_VALGRIND`, so that
  Valgrind sees every palloc chunk through PostgreSQL's own mempool annotations.

Each run started on a fresh copy of an `initdb -A trust --locale=C -E UTF8` cluster.

## Case 10, mmgr-repros 00 (`1f5b6a5e5d`, tuplestore)

| run | build | result |
|---|---|---|
| bug #19438's reproducer, `--single` | cassert | the expected `temp_file_limit` error, then `ROLLBACK` and the next statement complete; no assertion |
| `trigger.sql` (the reproducer, MOVE ALL for FETCH ALL), `--single`, Valgrind | Valgrind | one error, `Invalid free()` at `pfree` (mcxt.c:1531) by `tuplestore_end` (tuplestore.c:462) by `PortalDrop` by `CreatePortal` by `exec_simple_query`; the backend completes |
| the reproducer as reported, `--single`, Valgrind | Valgrind | the same error at the same frames |
| `control.sql` (FETCH BACKWARD 1000), `--single`, Valgrind | Valgrind | no error; 1000 rows, the backend completes |

The cassert build's silence is not evidence: `MEMORY_CONTEXT_CHECKING` does not catch a chunk freed
twice. The Valgrind build is the instrument, and the control shows it quiet when nothing is freed twice.

## The four that seemed not to run in the single-user backend (three remain)

| mmgr-repros case | fix | why not |
|---|---|---|
| 01 vacuum dead items | `3549ffb6af` | The stale pointer is left by a parallel vacuum with more than one index pass. Parallel workers are background workers, which the postmaster starts, and a single-user backend has no postmaster. Not run. |
| 02 live_parts | `ed394c4bdf` | Needs the only live partition dropped by another session between pruning and `try_table_open()` during planning. One backend cannot do that. Not run. |
| 03 pgoutput | `a61592253e` | Measured. On a 17.5 server, the thread's reproducer fails its first `pg_logical_slot_get_binary_changes` with "cannot use different column lists" and the second call kills the backend, natively. In `--single` both calls return 0 rows. pgoutput emits nothing there: with a plain `FOR TABLE` publication the same peek returns 4 rows on the server and 0 in `--single`, while test_decoding decodes the same insert in `--single`. The change callback never runs, so the error path that leaves the stale cache is never taken. |
| 04 reorderbuffer | `9e0b4b1ab5` | Runs, as case 11, added later the same day. See the next section. |

Why pgoutput emits nothing in single-user mode was not investigated.

## Case 11, mmgr-repros 04 (`9e0b4b1ab5`, reorderbuffer), found reachable later the same day

Upstream's reproduction recompiles `CHANGES_THRESHOLD` to 1. The release as shipped reaches the same
read when the hundredth change of a decoded transaction is a speculative insert's confirmation.
Transactions of 90 to 110 rows, each ending in an `INSERT ... ON CONFLICT`, decoded through
test_decoding in `postgres --single -c wal_level=logical` on the Valgrind build, gave exactly one
error. Then one transaction at a time:

| rows before the upsert | Valgrind |
|---|---|
| 97 | no error |
| 98 | `Invalid read of size 8` at `ReorderBufferProcessTXN` (reorderbuffer.c:2501), in a Slab block from `ReorderBufferGetChange` |
| 99 | no error |

98 is case 11's `trigger.sql`, and 99 its `control.sql`.
