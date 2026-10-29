# pwithbackport -- a real released SQLite UAF reverted into 3.22.0

## What it is

A use-after-free that shipped in **released SQLite 3.8.4 through 3.8.8**, reintroduced into the
3.22.0 engine by reverting its one-line upstream fix. It is a **backport**, labelled as such and
reported on its own line, never counted as a native 3.22.0 defect: stock 3.22.0 carries the fix and
is not vulnerable.

## The upstream bug and its fix

Upstream fix `f932f7145e` (2015-04-12, "Fix a problem with a stale Select.pWith pointer when a CTE is
used together with a compound query and an ORDER BY clause") added one line to
`convertCompoundSelectToSubquery()` in `src/select.c`:

```
  p->pNext = 0;
+ p->pWith = 0;
  p->selFlags &= ~SF_Compound;
```

The function rewrites a compound SELECT carrying an `ORDER BY ... COLLATE ...` into a subquery. It
shallow-copies the Select (`*pNew = *p`), so `pNew` and `p` share every pointer field, then nulls
each shared field on the original so the two do not both own it. Without the `pWith` line both
Selects own the same `With` object, freed on parse teardown
(`sqlite3SelectDelete -> clearSelect -> sqlite3WithDelete`) and then read again through the other
Select.

Released versions that shipped the bug (parent of the fix, lacking the fix): version-3.8.4,
3.8.4.1, 3.8.5, 3.8.6, 3.8.7, 3.8.7.1, 3.8.8. (`git tag --contains <fix>~1 --no-contains <fix>` on
the upstream history.)

## Why it belongs in the lookaside demonstration

The stale object is a `With` struct, `sqlite3DbMallocZero(db, ...)` with a live connection, about 48
bytes for one CTE, which is served from lookaside. The native proof (below) shows the pointer lies
inside the connection's lookaside arena. This is the region PoisonCap's `mem5.c` does not cover (it
acts only on memsys5 blocks; lookaside is one `sqlite3Malloc` block it never revokes inside), while
the Sublet memsys5 port issues and revokes each slot as its own child lifetime
(`sublet-3220000-memsys5.patch`: `lookasideIssue`/`lookasideRelease`). The constructed probes
`repro322/case_lookaside_uaf.c` and `case_lookaside_tagmap.c` (tags.tsv: probe) established the same
mechanism with a planted pattern; this case establishes it with a real released defect and a real
engine object through an ordinary SQL query.

## Trigger

Upstream's own regression test (`test/with1.test` 14.1):

```sql
CREATE TABLE t(a,b);
WITH x AS (SELECT * FROM t) SELECT 0 EXCEPT SELECT 0 ORDER BY 1 COLLATE binary;
```

No string literals, so `-DSQLITE_DQS=0` is irrelevant.

## Native proofs (host, before any platform run)

- lookaside **off**, ASan: `heap-use-after-free` in `sqlite3WithDelete`, freed via `sqlite3DbFreeNN`
  (the lookaside-aware free path). The bug is real on 3.22.0 with the line removed.
- lookaside **on**, diagnostic at the backport site: the stale `p->pWith` pointer lies inside
  `[db->lookaside.pStart, db->lookaside.pEnd)`, slot size 1200 -- the object is a slot -- and the
  process exits 0: the reuse is invisible to the allocator.

## Expected on the platforms

- PoisonCap (published, lookaside on): MISS -- the slot is not covered by `mem5.c`.
- Sublet (its memsys5 lookaside port, lookaside on): CATCH -- the slot is a revoked child lifetime,
  so the stale read after `sqlite3DbFreeNN` faults (cause 25).
