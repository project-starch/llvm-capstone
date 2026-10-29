# 2026-10-29: mmgr-repros' five live defects, asked to run whole

`../../../mmgr-repros` holds five defects live at 17.5 as C programs against the real memory
contexts. This run asked of each: does upstream's own trigger reach it in the whole backend, as the
Capstone port runs it -- `postgres --single`, one process, no postmaster? One does, and is case 10.

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

## The four that cannot run in the single-user backend

| mmgr-repros case | fix | why not |
|---|---|---|
| 01 vacuum dead items | `3549ffb6af` | The stale pointer is left by a parallel vacuum with more than one index pass. Parallel workers are background workers, which the postmaster starts, and a single-user backend has no postmaster. Not run. |
| 02 live_parts | `ed394c4bdf` | Needs the only live partition dropped by another session between pruning and `try_table_open()` during planning. One backend cannot do that. Not run. |
| 03 pgoutput | `a61592253e` | Measured. On a 17.5 server, the thread's reproducer fails its first `pg_logical_slot_get_binary_changes` with "cannot use different column lists" and the second call kills the backend, natively. In `--single` both calls return 0 rows. pgoutput emits nothing there: with a plain `FOR TABLE` publication the same peek returns 4 rows on the server and 0 in `--single`, while test_decoding decodes the same insert in `--single`. The change callback never runs, so the error path that leaves the stale cache is never taken. |
| 04 reorderbuffer | `9e0b4b1ab5` | Upstream reproduces it only after hardcoding `CHANGES_THRESHOLD` to 1 and running test_decoding's `ddl` test under Valgrind. The release as shipped needs a hundred changes inside one speculative insert's window, and no trigger for that is published. |

Why pgoutput emits nothing in single-user mode was not investigated.
