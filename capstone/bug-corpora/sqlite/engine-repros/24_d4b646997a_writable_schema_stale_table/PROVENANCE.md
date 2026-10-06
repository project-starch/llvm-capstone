# wschema -- d4b646997a

Upstream fix `d4b646997a`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row8 / sqlite-d4b646997a -- sqlite3StartTable() records the table in
 * pSchema->pSeqTab BEFORE the table is finalized:
 *
 *     if( !pParse->nested && strcmp(zName, "sqlite_sequence")==0 ){
 *       pTable->pSchema->pSeqTab = pTable;
 *     }
 *
 * If the CREATE then fails, sqlite3Prepare's cleanup does
 * sqlite3DeleteTable(db, sParse.pNewTable) and the Table is freed, but
 * pSchema->pSeqTab still points at it. Later AUTOINCREMENT codegen dereferences it:
 *
 *     autoIncBegin: sqlite3OpenTable(pParse, 0, p->iDb, pDb->pSchema->pSeqTab, OP_OpenRead)
 *
 * Fixed 3.36.0. Core.
 *
 * Reaching it needs an explicit "CREATE TABLE sqlite_sequence", which
 * sqlite3CheckObjectName normally rejects -- but only when
 * (db->flags & SQLITE_WriteSchema)==0, so PRAGMA writable_schema=1 is the opener.
 * The statement must then fail AFTER StartTable; a duplicate column name does it
 * (sqlite3AddColumn -> "duplicate column name"), which runs after the pSeqTab store.
 *
 * NOTE this does NOT need a corrupt database file, contrary to the first assessment
 * of this row: writable_schema plus a post-StartTable error is sufficient, and the
 * domain is :memory:-only.
 * NOTE -DSQLITE_DQS=0 -> SQL string literals must be single-quoted.
```
