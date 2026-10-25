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
