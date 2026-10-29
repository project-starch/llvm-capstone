# `1f5b6a5e5d` -- a tuplestore's tuples freed twice when a spill fails

A double free in PostgreSQL's tuplestore, reached through SQL against the whole backend.

## The defect

`dumptuples()` writes each in-memory tuple to a temporary file and frees it as it goes
(`WRITETUP` is `writetup_heap`, which ends with `heap_free_minimal_tuple`). It clears
`memtupdeleted` and `memtupcount` only after the loop. When `BufFileWrite` throws part way --
here because `temp_file_limit` is exceeded -- the counters still describe every slot as live while
the first ones point at freed chunks. The `ROLLBACK` drops the unnamed portal that held the FETCH's
result, and `tuplestore_end()` walks those slots and frees them a second time.

## Upstream defect

Upstream fix `1f5b6a5e5d`, 2026-03-30, "Be more careful to preserve consistency of a tuplestore",
bug #19438. The fix postdates the 17.5 pin and is not in `REL_17_5`
(`git merge-base --is-ancestor 1f5b6a5e5d REL_17_5` is false). The commit message says the
submitted test "failed to fail" on v16 and v17, and the fix omits it. Natively that is what happens:
the second `pfree` returns a chunk to the context's free list twice and the backend carries on.

- **CVE:** `NO VERIFIED CVE`.
- **Sibling:** `../../mmgr-repros/00_1f5b6a5e5d_tuplestore_double_free` is the same defect as a C
  program against the real memory contexts, with a labelled probe. Not two defects.

## The trigger

Bug #19438's reproducer, run through `postgres --single`, with one change. The report discards
the forward `FETCH ALL`'s million rows with psql's `\o /dev/null`, which a single-user backend does
not have, so the cursor is moved to its end with `MOVE ALL` instead.

## Liveness at the pin, measured

PostgreSQL 17.5 from the port's tarball, built `--enable-debug CFLAGS='-O1 -g'
CPPFLAGS=-DUSE_VALGRIND`, without assertions, as the port is built. `trigger.sql` run through
`postgres --single` under Valgrind 3.26 with `src/tools/valgrind.supp` gives one error:

    Invalid free() / delete / delete[] / realloc()
       at pfree (mcxt.c:1531)
       by tuplestore_end (tuplestore.c:462)
       by PortalDrop (portalmem.c:585)
       by CreatePortal (portalmem.c:193)
       by exec_simple_query

That is the report's own backtrace, on the `ROLLBACK`. `control.sql` gives no error on the same
build. The backend answers `survived` in both: natively the double free is silent.

## Negative control

`control.sql` is the same statements with `FETCH BACKWARD 1000` in place of `FETCH BACKWARD ALL`.
A thousand rows fit in memory under the same limit, so nothing spills and each tuple is freed once.
