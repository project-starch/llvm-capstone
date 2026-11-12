# `9e0b4b1ab5` -- logical decoding reads the LSN of a change it just freed

A stale read in PostgreSQL's reorder buffer, reached through SQL against the whole backend.

## The defect

`ReorderBufferProcessTXN()` replays a transaction's changes to the output plugin and, after every
`CHANGES_THRESHOLD` (100) of them, reports progress with `change->lsn`. An `INSERT ... ON CONFLICT`
decodes as a speculative insert, held, and a confirmation. Processing the confirmation sends the held
insert and returns it to the Change context's Slab. When that confirmation is the hundredth change,
the progress update reads the LSN of the change just freed.

## Upstream defect

Upstream fix `9e0b4b1ab5`, 2025-08-02, "Fix use-after-free with INSERT ON CONFLICT changes in
reorderbuffer.c", back-patched to 16. It postdates the 17.5 pin and is not in `REL_17_5`. The
commit message says the reproduction was CHANGES_THRESHOLD hardcoded at 1 and test_decoding's
`ddl` test under Valgrind.

- **CVE:** `NO VERIFIED CVE`.
- **Sibling:** `../../mmgr-repros/04_9e0b4b1ab5_reorderbuffer_spec_insert_lsn`, the same defect as a
  C program against the real Slab context. Not two defects.

## The trigger

The threshold is not recompiled. The transaction is sized instead. 98 rows are inserted, then one
row by `INSERT ... ON CONFLICT`, so the speculative insert is change 99 and its confirmation change
100. The transaction is then decoded through test_decoding. Logical decoding needs
`wal_level=logical`, which SQL cannot set, so the file asks for it with `-- SERVER-SETTING:`
(`shared/run-arm.py`). The port links test_decoding into the image as an output plugin
(`ports/postgres/app/build-domain.sh`).

## Liveness at the pin, measured

PostgreSQL 17.5 built without assertions and with `USE_VALGRIND`, test_decoding installed, the
trigger's statements through `postgres --single -c wal_level=logical` under Valgrind 3.26:

    Invalid read of size 8
       at ReorderBufferProcessTXN (reorderbuffer.c:2501)
       by ReorderBufferReplay (reorderbuffer.c:2730)
       by ReorderBufferCommit (reorderbuffer.c:2754)
       by DecodeCommit (decode.c:746)
       ... by pg_logical_slot_get_changes (logicalfuncs.c:333)
     Address ... is 2,144 bytes inside a recently re-allocated block of size 8,192 alloc'd
       at malloc / SlabAllocFromNewBlock (slab.c:565) / SlabAlloc / MemoryContextAlloc
       by ReorderBufferGetChange (reorderbuffer.c:490)

With 97 or 99 rows there is no error, as the arithmetic predicts. Natively the read is silent and
decoding returns its 101 rows.

## Negative control

`control.sql` runs the same statements with 99 rows. The hundredth change is then the speculative
insert itself, still held, so the progress update reads a live change.
