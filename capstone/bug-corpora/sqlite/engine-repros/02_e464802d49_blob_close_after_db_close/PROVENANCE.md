# blobclose -- e464802d49

Upstream fix `e464802d49`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
sqlite_blobclose_domain.c -- CONTROL (unprotected) reproduction of upstream
 * SQLite fix e464802d49: sqlite3_blob_close() uses a lookaside-allocated Incrblob
 * after sqlite3_close_v2() has freed its connection and the lookaside pool.
 * Public, already-fixed bug (fixed in 3.29.0); collected and reproduced for the
 * Capstone/Sublet temporal-safety study.
 *
 * CONTROL arm: SQLite's own memsys5 heap, lookaside ON, nothing revoked. On
 * unprotected Capstone the post-free access is NOT caught, so the domain RETURNS;
 * the matching host ASan build is what shows the heap-use-after-free.
```

## Reviewed 2026-10-15

The defect, the trigger and all three verdicts stand. What changed is three claims in
`case.json` that the case's own source already contradicted, and the divergence between the
two copies of the trigger.

**The freed object is the connection, not the Incrblob.** `sqlite3_blob_close` finalizes the
blob's statement first; that is the last statement on a connection `sqlite3_close_v2` has
already made a zombie, so `sqlite3LeaveMutexAndCloseZombie` frees the connection. The same
function then calls `sqlite3DbFree(db, p)` to release the Incrblob, and `sqlite3DbFreeNN` reads
`db->lookaside` on the connection it has just freed. Host ASan: 696 bytes inside a 704-byte
region allocated by `openDatabase`. `case.json` had named the Incrblob as the object and
`sqlite3_close_v2` as what ended its lifetime; both are now the connection and the finalize.

**Lookaside is off in this corpus**, which the `fidelity` field said the opposite of. The build
sets `-DSQLITE_DEFAULT_LOOKASIDE=0,0`, the case prints `sqlite3_db_status` to show
`slots_in_use=0`, and the stale read is of the connection's lookaside *descriptor*, which
`sqlite3DbFree` consults whether or not a pool exists. `fidelity` also said the Sublet arm
skips this case; that was true of the legacy standalone domain and has not been true since the
2026-10-05 port onto `REPRO322_MAIN`.

**Evidence** (result lines; probe sites 2 and 39 in `shared/probe/`):

- Host ASan reproduces the 2026-10-11 report exactly.
- Site 39 (the `sqlite3_free(db)` in `sqlite3LeaveMutexAndCloseZombie`) prints the connection it
  frees; site 2 (`db->pnBytesFreed` in `sqlite3DbFreeNN`, the line host ASan reports) gives
  `kind=UAF off=0 n=8` on hit 155 of 155 -- the preceding 154 are ordinary frees, so the witness
  is the defect and not the traffic. With the connection closed by plain `sqlite3_close` instead
  of `close_v2`, which returns SQLITE_BUSY and frees nothing, site 39 prints nothing and site 2
  has no witness in 144 hits.
- Measured 2026-10-15: virtual-malloc MISSED, virtual-nested-pools CAUGHT with cause 25, the
  lifetime check, in `sqlite3DbFreeNN`, both arms' controls as declared; CheriBSD purecap
  `RETURNED rc=0` with its positive control faulting under revocation. On CheriBSD the probe
  adds why: the pointer the read goes through carries `tag=1 cb=<arena>+262144`, the bounds of
  the whole 256 KiB memsys5 arena, and the access is inside them.

**The two copies of the trigger are now one file.** `ports/sqlite/cheribsd/cases/case_blobclose.c`
had been rewritten from the same legacy domain separately from this one: same API sequence, same
SQL, differing only in a local `fail()` helper where this copy uses the harness's `FAILRC`, and
in `(unsigned)` against `(unsigned long)` casts. The CheriBSD harness defines `FAILRC` too, and
14 of its other cases already use it, so that copy is now this file byte for byte. Neither
reason `ports/sqlite/cheribsd/port.json` gives for the two sets differing applies here: this
case opens `:memory:` and needs no database file, and the plain build carries no probe.
