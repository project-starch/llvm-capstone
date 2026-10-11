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
