# staticbind -- eab0e10304

Upstream fix `eab0e10304`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
new-3 / sqlite-eab0e10304 -- fts3/fts5/rtree leave a freed heap buffer bound
 * SQLITE_STATIC to a PERSISTENT cached statement.
 *
 * fts3's segment writer binds its own buffer into the cached segdir statement and
 * then frees it, with no bind_null in between:
 *     sqlite3_bind_blob(pStmt, 6, zRoot, nRoot, SQLITE_STATIC);
 *     sqlite3_step(pStmt);
 *     rc = sqlite3_reset(pStmt);
 * sqlite3_reset() does NOT clear bindings, so parameter 6 keeps pointing at the
 * buffer after fts3SegWriterFree() releases it. Fixed 3.23.0 -- 16 days after
 * 3.22.0 shipped -- by adding the bind_null. The fix's own commit message states
 * the threat model: "a user may obtain a pointer to the persistent statement using
 * sqlite3_next_stmt() and attempt to access the freed buffer using
 * sqlite3_expanded_sql() or similar".
 *
 * WHY THIS DOMAIN DOES NOT USE sqlite3_expanded_sql().
 * It cannot: this port compiles with -DSQLITE_OMIT_FLOATING_POINT=1, and 3.22.0's
 * sqliteInt.h reacts to that with
 *     #define SQLITE_OMIT_DATETIME_FUNCS 1
 *     #define SQLITE_OMIT_TRACE 1
 * so sqlite3_expanded_sql() is compiled as `return 0;` in EVERY group of this port.
 * A command-line -USQLITE_OMIT_TRACE does not help, because the #define happens
 * inside the translation unit. (This was verified with a dedicated probe: every
 * expanded_sql call returns NULL even for a no-parameter statement in the smallest
 * image with a completely idle arena, and the arena serves all size classes from 64
 * to 16384 bytes. An earlier session blamed memsys5 fragmentation; that was wrong.)
 *
 * So this domain takes the "or similar" route the fix names: re-STEP the cached
 * statement without rebinding. Parameter 6 still points at the freed buffer, so
 * OP_MakeRecord reads it and REPLACEs the %_segdir row with whatever is there now.
 *
 * Host ASan oracle on 3.22.0, same shape:
 *   re-step  : READ of size 1492, 0 bytes into a freed 4064-byte region
 *                use   sqlite3VdbeSerialPut <- sqlite3VdbeExec <- sqlite3_step
 *   expanded : READ of size 1    (upstream's route, for comparison)
 *   free  fts3SegWriterFree <- fts3SegmentMerge <- fts3DoOptimize
 *           <- fts3SpecialInsert <- sqlite3Fts3UpdateMethod
 *   alloc fts3SegWriterAdd <- fts3SegmentMerge
 *
 * This route is also OBSERVABLE without any memory-safety tool, which matters
 * because on base Capstone the read is silent (the buffer is read as bytes, not as
 * a pointer). On the host the re-step rewrites the row with DIFFERENT content at the
 * same length, and the index is corrupt afterwards:
 *     before len=1492 sum=3696465335 first4=0006616c
 *     after  len=1492 sum=951463155  first4=60ed69d4   <- looks like a heap pointer
 *     integrity-check -> SQLITE_CORRUPT
 * In the domain memsys5 writes its in-band Mem5Link freelist ints over the start of
 * the freed block, so a changed hash here is the freed block's new contents being
 * copied into the database.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```

## Review 2026-10-16 -- every verdict holds, and this is the pair's better-made half

### The four readings

| reading | result |
|---|---|
| host ASan | `heap-use-after-free`, READ of size 1492, 0 bytes into a freed 4064-byte region, `sqlite3VdbeSerialPut <- sqlite3VdbeExec <- sqlite3_step`; freed by `fts3SegWriterFree <- fts3SegmentMerge <- fts3DoOptimize`, allocated by `fts3SegWriterAdd <- fts3SegmentMerge`. 2026-10-11 bundle; not re-run, this host has no x86 ASan runtime |
| virtual-malloc | **MISSED**, `RETURNED`, controls as declared |
| virtual-nested-pools | **CAUGHT**, `cause=25 pc=0x3fb18020f8` offset 204 in `memcpy`, controls as declared |
| cheribsd-revocation | **MISSED**, `RETURNED rc=0`, positive control trapping, revocation knobs 1/0 |

The mechanism is confirmed in the pinned amalgamation: `fts3WriteSegdir` does
`sqlite3_bind_blob(pStmt, 6, zRoot, nRoot, SQLITE_STATIC)` at `sqlite3.c:160853`, steps and
resets, and `sqlite3_reset()` does not clear bindings. `zRoot` on the path this trigger takes is
`pWriter->aData` itself -- `fts3SegWriterFlush`'s else branch passes it directly when the whole
tree fits on the root node -- and `fts3SegWriterFree` then does `sqlite3_free(pWriter->aData)`.
That is the 4064-byte region host ASan reports, allocated in `fts3SegWriterAdd`.

### The attribution, now pointer identity

`fault_sites_why` admitted that any `memcpy` under this case would count, and that it was
declared after the runs. Probe site 58 hooks the bind, the free, and the copy in
`sqlite3VdbeSerialPut` (filtered to `len>400`, so the root buffer separates from that hot
function's ordinary traffic). On CheriBSD purecap:

```
SERIAL  z=0x218200 len=1665 first4=0x7616c       <- the legitimate write, before the free
FREE    aData=0x218200 nData=1665 (segdir param 6 may still point here)
staticbind before re-step
LB_WITNESS kind=UAF off=0 n=1665 req=4061 blk=0x218200 tag=1 cb=0x206200+262144 ca=0x218200
SERIAL  z=0x218200 len=1665 first4=0x40050000    <- the stale read: same pointer, changed content
```

One pointer value at all three hooks, and the length at the copy is the `nData` at the free. So
the `memcpy` the Sublet arm faults in is the one copying out of the freed root buffer.

### The one case so far where the address judgement is correct and sufficient

`kind=UAF off=0 n=1665 req=4061` on block `0x218200`: the block is **not** reissued before the
read, `req=4061` matches the 4064-byte region host ASan reports, and the 1665-byte read sits
inside it. That is not the `kind=OOB` misclassification cases 09, 17 and 18 produced, where a
reclaimed block made the read exceed the new owner's request. Here the probe's witness is
evidence on its own, and `witnesses=1 first_witness_at_hit=6` of 7 hits localises it exactly.

It also shows why this case produces **no false positive** on the two non-Sublet arms, unlike
cases 15, 16 and 18. At the read the capability has `tag=1` and bounds covering the whole
256 KiB arena, so neither architecture has anything to object to; and what is read is **bytes**,
copied into a database row, not a pointer that is then dereferenced. No forged pointer is ever
created, so the baseline tag check never fires. The case's own header anticipated this -- "on
base Capstone the read is silent (the buffer is read as bytes, not as a pointer)" -- and it is
the only header in this corpus that did.

### The consequence, and why integrity_check is not the signal

The re-step rewrites the `%_segdir` row at the same length with different content, on every arm.
The bytes that land there differ by platform, all of them read out of the same dead buffer:

| platform | `first4` after the re-step | `PRAGMA integrity_check` |
|---|---|---|
| virtual-malloc | `0xFFFFFFFF` -- memsys5's own `Mem5Link` `next=-1` terminator | `SQLITE_NOMEM` |
| cheribsd-revocation | `0x40050000` | `SQLITE_CORRUPT` |
| host (recorded) | `0x60ed69d4`, "looks like a heap pointer" | `SQLITE_CORRUPT` |

`integrity_check` returning `SQLITE_NOMEM` on base Capstone is not a clean result: the 256 KiB
arena cannot complete the check. So the signal to report is the changed row -- `len` unchanged,
`sum` and `first4` changed -- and not `integrity_check`, which is inconclusive on one of the
three arms. Where it does run, it says `SQLITE_CORRUPT`, so unlike case 17's rtree shadow tables
this corruption is detectable by SQLite itself; just not by the memory-safety mechanism.

### Records corrected

* `fidelity` said "same upstream fix as case 18". Case 22 is `fix-2026-07-26` and unrelated; the
  pair sharing `eab0e10304` is **21** (rtree) and **23** (fts3). Case 21's own `fidelity` said
  "case 20", which is `d4b646997a`. The renumber from 38 cases to 33 on 2026-10-06 left both
  halves of the pair pointing at the wrong row; both are fixed, and case 13's review had already
  found one of these.
* `fault_sites_why` records the identity evidence in place of "any memcpy would count".
* The physical `sublet` row is marked: it reports `cause 24`, which cannot establish that the
  lifetime check fired, so the Sublet cell rests on the virtual arm's `cause 25`.
* `run-probe-cheri.sh`'s SITE entry for `staticbind` moves from the arm's old site 4 to 58.
* Unlike case 17, this case's corpus `case.c` and `ports/sqlite/cheribsd/cases/` copy were
  **already byte-identical** (202 lines each), so nothing had to be unified.
