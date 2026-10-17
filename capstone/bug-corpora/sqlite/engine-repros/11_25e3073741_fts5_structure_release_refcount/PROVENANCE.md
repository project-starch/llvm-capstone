# fts5structwrite -- 25e3073741

Upstream fix `25e3073741`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row3 / sqlite-25e3073741 -- fts5MultiIterNew caches a raw Fts5Structure*; a
 * table write under an active cursor releases it, dangling the pointer;
 * fts5MultiIterFree releases it again (fts5_index.c). Fixed 3.27.0. CONTROL:
 * open a scan cursor, INSERT into the same table while the cursor is live, keep
 * stepping. On unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts5 group (repro322_fts_stubs.c + math decl).
```

## 2026-10-11: the trigger is the CheriBSD copy

`case.c` is now byte-identical to `ports/sqlite/cheribsd/cases/` (the copy the CheriBSD probe runs used).
The source above is the earlier trigger. Why it was replaced: the corpus copy inserted under a three-term OR scan; host ASan was silent on it. The CheriBSD copy uses upstream's shape (x3, MATCH 'one', an INSERT under the live cursor), and host ASan reports the second release in fts5StructureRelease <- fts5MultiIterFree. Evidence:
`results/2026-10-11-host-asan/results.json`, from `shared/run-host-asan.py` (every SQLite object its own
malloc, lookaside off).

## 2026-10-16: re-measured; why the unprotected arms are silent is not what the case said

Every verdict holds. The trigger, corrected on 2026-10-11 to upstream's fts5update.test 3.0/3.1
shape, reaches the defect on this platform and its own reachability contract held in every run:
3 rows and 3 successful in-loop writes.

Host ASan:

```
READ of size 4, 0 bytes inside a freed 40-byte region   -- offset 0 is Fts5Structure.nRef
  use   fts5StructureRelease sqlite3.c:195026 <- fts5MultiIterFree <- sqlite3Fts5IterClose
          <- fts5ExprNearInitAll <- fts5ExprNodeFirst <- sqlite3Fts5ExprFirst
          <- fts5CursorReseek <- fts5NextMethod
```

`fts5StructureRelease` is `if( pStruct && 0>=(--pStruct->nRef) )`, so the stale operation is a read
**and a decrement**, with a second free if the result reaches zero.

**The correction.** The case's header said nRef at offset 0 is exactly the granule memsys5's
8-byte `Mem5Link` overwrites, so in the domain the refcount read returns a freelist integer rather
than faulting. Probe site 50 -- the caching store in `fts5MultiIterNew`, every release with the
`nRef` value it reads, and the free -- says otherwise, identically on native memsys5, on virtual
Capstone with lookaside off and on CheriBSD purecap:

```
31 release hits, witnesses=0 on all three platforms    -- the block is never free when read
trace for the busiest structure:
  CACHE    into Fts5Iter.pStruct
  RELEASE  nRef_read=3 / 3 / 2 / 1
  FREE
  CACHE    same address, into Fts5Iter.pStruct         -- reissued
  RELEASE  nRef_read=3 / 2                             -- a fresh refcount: a new tenant
```

memsys5 reissues the block to the next `Fts5Structure` -- the same size class -- before the stale
release arrives. So the stale `--nRef` does not read allocator bookkeeping; it silently decrements
**a live successor object's** refcount. The `Mem5Link` would only be visible if the block were still
free at the read, which it never is in these runs.

That makes this case a sharper illustration than the header claimed. The memory the stale operation
touches is not garbage that a sanity check could reject -- it is another object's valid data, with a
plausible refcount. No value-based oracle and no address-based oracle can see it; only identity can,
and Sublet's cause 25 is the arm that does. On CheriBSD, fts5's own `integrity-check` returns rc=0
afterwards, so the corruption is invisible to the engine as well.

Arms, measured 2026-10-16 on this platform:

| arm | verdict | evidence |
| --- | --- | --- |
| host ASan | reports | as above |
| virtual-malloc | MISSED | both controls as declared; rows=3, in_loop_writes_ok=3, RETURNED |
| virtual-nested-pools | CAUGHT | both controls as declared; `cause=25 pc=0x3f8a21bf40 in fts5StructureRelease`, attributed by function |
| cheribsd-revocation | MISSED | positive control traps; rows=3, integrity-check rc=0, RETURNED rc=0 |

No count in the study's tables moves. This case's Sublet cell was already detected, but it rested on
one of the 28 halts at kernel pc 0x800239e4 and now rests on a cause-25 fault at the declared site;
that physical row is marked in `results/matrix.tsv`.
