# fts5rank -- 12439f9c16

Upstream fix `12439f9c16`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row12 / sqlite-12439f9c16 -- fts5FilterMethod() parses rank from a config that
 * has NOT yet been reloaded. fts5CursorParseRank ALIASES the config strings:
 *     pCsr->zRank     = (char*)pConfig->zRank;
 *     pCsr->zRankArgs = (char*)pConfig->zRankArgs;
 * When another connection redefines rank, %_config changes and the cookie bumps;
 * the later sqlite3Fts5ConfigLoad does
 *     sqlite3_free(pConfig->zRank); sqlite3_free(pConfig->zRankArgs);
 * so the cursor's aliases dangle and the subsequent reads of pCsr->zRank are a
 * use-after-free. Fixed 3.43.2 (the fix hoists the reload above rank parsing).
 *
 * Needs TWO connections on ONE database, so this builds in the fts5S group with
 * -USQLITE_OMIT_SHARED_CACHE and opens file::memory:?cache=shared via
 * SQLITE_OPEN_URI. The VFS xOpen always returns SQLITE_CANTOPEN, but a
 * :memory: database (including the shared-cache URI form) takes the memDb path
 * and never calls xOpen.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```
