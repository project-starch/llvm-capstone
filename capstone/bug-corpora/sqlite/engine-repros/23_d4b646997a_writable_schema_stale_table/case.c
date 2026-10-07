/* row8 / sqlite-d4b646997a -- sqlite3StartTable() records the table in
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
 * NOTE -DSQLITE_DQS=0 -> SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void step(sqlite3 *db, const char *sql, const char *tag){
  char *e = 0;
  int rc = sqlite3_exec(db, sql, 0, 0, &e);
  out_text("wschema "); out_text(tag); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc));
  if (e){ out_text(" ("); out_text(e); out_text(")"); sqlite3_free(e); }
  out_text("\n");
}

static int run_case(void){
  if (repro_init()) return 1;

  /* CONTROL phase: the same AUTOINCREMENT create+insert on a clean connection, with
   * no dangling pSeqTab. This is what makes the poisoned phase below attributable:
   * if the control INSERT succeeds and only the poisoned one reports corruption,
   * the freed-pSeqTab read is what caused it. */
  {
    sqlite3 *c = 0;
    if (sqlite3_open(":memory:", &c) == SQLITE_OK){
      step(c, "CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT, y);", "CONTROL create-autoinc");
      step(c, "INSERT INTO t(y) VALUES(1);", "CONTROL insert (expect rc=0)");
    }
    sqlite3_close(c);
  }

  sqlite3 *db = 0; int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) return FAILRC("open", rc);

  step(db, "PRAGMA writable_schema=1;", "writable-on");

  /* CheckObjectName lets sqlite_ through; StartTable stores pSeqTab = pTable;
   * the duplicate column then aborts the statement and the Table is freed. */
  step(db, "CREATE TABLE sqlite_sequence(a, a);", "create-seq (expect dup-column error)");

  step(db, "PRAGMA writable_schema=0;", "writable-off");

  /* pSeqTab is non-NULL (dangling), so EndTable's "if( pDb->pSchema->pSeqTab==0 )"
   * skips creating a real sqlite_sequence for this AUTOINCREMENT table. */
  step(db, "CREATE TABLE t(x INTEGER PRIMARY KEY AUTOINCREMENT, y);", "create-autoinc");

  out_text("wschema before insert (autoIncBegin reads the freed pSeqTab)\n");
  step(db, "INSERT INTO t(y) VALUES(1);", "insert");

  sqlite3_close(db);
  out_text("wschema NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("wschema")
