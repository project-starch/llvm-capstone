# backupattach -- b9e0f62c3a

Upstream fix `b9e0f62c3a`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row15 / sqlite-b9e0f62c3a -- sqlite3_backup caches raw Btree* for src/dest
 * (backup.c:23/29, resolved by findBtree at 82). ATTACH reallocates db->aDb
 * (attach.c:113 sqlite3DbRealloc) and can invalidate the cached Btree*, so a
 * later backup step uses a freed/moved Btree. Fixed 3.53.3 (store db indices,
 * re-derive the Btree each step).
 *
 * CONTROL arm: start a backup, then ATTACH a database on the SAME connection to
 * force db->aDb realloc, then step the backup -- it dereferences the stale
 * Btree*. On unprotected Capstone the moved-Btree access is not caught, so the
 * step returns and the domain RETURNS. Host ASan flags heap-use-after-free.
```
