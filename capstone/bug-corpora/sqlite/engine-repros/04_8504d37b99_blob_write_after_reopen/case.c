/* row20 / sqlite-8504d37b99 -- sqlite3_blob_write()/read reads a released page
 * through an invalidated cursor. blobReadWrite() (vdbeblob.c:372) calls
 * sqlite3BtreeIntegerKey() (420) with no cursor-valid check, then the preupdate
 * hook (421) runs with that key. If the blob cursor was saved (another writer
 * modified/deleted the row), the page was released but pCur->pPage left set, so
 * the cell is parsed from a released page. Needs -DSQLITE_ENABLE_PREUPDATE_HOOK.
 * Fixed 3.51.0 (reseek first, skip hook if it cannot restore).
 *
 * CONTROL arm: open a blob, invalidate its cursor via another statement's write,
 * then blob_write -> IntegerKey on the released page. On unprotected Capstone the
 * released-page read is not caught, so it returns and the domain RETURNS.
 */
#include "repro322_common.h"

static void preupdate_cb(void *pctx, sqlite3 *db, int op, char const *zDb,
                         char const *zTbl, sqlite3_int64 k1, sqlite3_int64 k2) {
  (void)pctx;(void)db;(void)op;(void)zDb;(void)zTbl;(void)k1;(void)k2;
}

static int run_case(void) {
  if (repro_init()) return 1;
  sqlite3 *db = 0;
  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) return FAILRC("open", rc);

  sqlite3_preupdate_hook(db, preupdate_cb, 0);

  rc = sqlite3_exec(db,
    "CREATE TABLE t(x);"
    "INSERT INTO t VALUES(zeroblob(64));"      /* rowid 1, the blob row */
    "INSERT INTO t VALUES(zeroblob(64));",     /* rowid 2, gives the tree >1 cell */
    0,0,0);
  if (rc != SQLITE_OK) return FAILRC("setup", rc);

  sqlite3_blob *blob = 0;
  rc = sqlite3_blob_open(db, "main", "t", "x", 1, 1 /*rw*/, &blob);
  if (rc != SQLITE_OK) return FAILRC("blob-open", rc);

  /* Another writer on the SAME connection: delete/insert to force the blob's
   * b-tree cursor to be SAVED (page released, pCur->pPage stale). */
  rc = sqlite3_exec(db, "INSERT INTO t VALUES(zeroblob(4000));"
                        "DELETE FROM t WHERE rowid=2;", 0,0,0);
  if (rc != SQLITE_OK) { out_text("blobwrite writer rc="); out_uint((unsigned)rc); out_text("\n"); }

  /* blobReadWrite: IntegerKey through the invalidated cursor + preupdate hook. */
  static const char zeros[16] = {0};
  rc = sqlite3_blob_write(blob, zeros, 16, 0);
  out_text("blobwrite write rc="); out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");

  sqlite3_blob_close(blob);
  sqlite3_close(db);
  out_text("blobwrite NOTRAP done\n");
  return 0;
}

REPRO322_MAIN("blobwrite")
