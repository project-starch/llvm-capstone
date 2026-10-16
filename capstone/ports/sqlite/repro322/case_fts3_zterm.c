/* row11 / sqlite-dee0359ddb -- fts3SegReaderNext() aliases pReader->zTerm straight into
 * the pending-terms hash key memory (fts3_write.c:1333) while copying the doclist beside
 * it; a PendingTermsClear during the scan frees that key, and the next term comparison
 * reads it. Fixed 3.37.0.
 *
 * WHY THIS SHAPE. Three ingredients, each necessary, found by measurement 2026-10-16:
 *   1. The inserts must be in an UNCOMMITTED transaction. Every commit runs
 *      sqlite3Fts3PendingTermsFlush, which clears the hash, so with auto-commit the
 *      pending hash is already empty when the scan opens and no zTerm ever aliases it.
 *      That was this case's trigger until today.
 *   2. Some rows must be COMMITTED first, so the phrase's cursor holds a segment reader
 *      AND a pending reader. zTerm is only read when terms are compared BETWEEN readers
 *      (fts3SegReaderCmp); the per-row advance, sqlite3Fts3MsrIncrNext, compares docids
 *      only, and for a pending reader it even copies the position list into pMsr->aBuffer
 *      first. Upstream copies everything on this path except the term.
 *   3. After the clear there must be a RESTART, because fts3EvalRestart ->
 *      fts3EvalPhraseStart -> fts3EvalPhraseLoad -> fts3TermSelect steps the EXISTING
 *      Fts3MultiSegReader again, and that is what compares the terms. The restart is
 *      reached the way case 08 reaches it: offsets() over a NEAR group under an OR whose
 *      AND operand to the left fails first, so fts3EvalTestExpr short-circuits before it
 *      can zero the group.
 *
 * Host ASan oracle (prepared 3.22.0 amalgamation, system allocator, lookaside off):
 *   READ of size 1, 0 bytes inside a freed 8-byte region
 *     use   fts3SegReaderCmp <- fts3SegReaderSort <- fts3SegReaderStart
 *             <- fts3TermSelect <- fts3EvalPhraseLoad <- fts3EvalPhraseStart
 *             <- fts3EvalRestart <- fts3EvalRestart <- sqlite3Fts3EvalPhrasePoslist
 *             <- fts3ExprTermOffsetInit <- sqlite3Fts3Offsets
 *     free  fts3HashFree <- sqlite3Fts3HashClear <- sqlite3Fts3PendingTermsClear
 *             <- fts3DoOptimize <- fts3SpecialInsert
 *     alloc fts3HashMalloc <- sqlite3Fts3HashInsert <- fts3PendingTermsAddOne
 *             <- fts3InsertTerms -- the hash key copy, which is the aliased object
 * The access is inside the restart, so it strictly precedes the stale read case 08
 * reports in sqlite3Fts3Offsets: the two cases do not mask each other, and this one's
 * fault_sites is fts3SegReaderCmp.
 *
 * THIS CASE NEEDS ITS OWN GROUP: the nesting and the AND operator exist only with
 * -DSQLITE_ENABLE_FTS3_PARENTHESIS (group fts3P). Without it the parentheses and AND are
 * ordinary tokens, the query matches nothing, and the case says so.
 * REACHABILITY PROBE: row 1's offsets() must be exactly ROW1_OFFSETS, and the mid-scan
 * optimize must return 0. The stale read is inside the SECOND row's offsets() call.
 * CONTROL: on unprotected capstone the comparison completes -> NOTRAP.
 * NOTE: build in fts3P group. -DSQLITE_DQS=0, so SQL literals are single-quoted. */
#include "repro322_common.h"

/* row 1 is 'apple xx qq segone': apple@0, xx@6, qq@9, and the term indices are the
 * phrase order of the query. Fixed by the document text, so it is allocator-independent. */
#define ROW1_OFFSETS "0 1 0 5 0 0 6 2 0 2 9 2"

static int streq(const char *a, const char *b){
  if(!a || !b) return 0;
  while(*a && *a==*b){ a++; b++; }
  return *a==0 && *b==0;
}

static int X(sqlite3 *db, const char *label, const char *sql){
  int rc = sqlite3_exec(db, sql, 0, 0, 0);
  if(rc){
    out_text("fts3zterm ");out_text(label);out_text(" rc=");out_uint((unsigned)rc);
    out_text(" (");out_text(sqlite3_errmsg(db));out_text(")\n");
  }
  return rc;
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);

  if(X(db,"create","CREATE VIRTUAL TABLE ft USING fts3(a);")) { sqlite3_close(db); return 1; }
  /* committed, so these terms live in a segment: the cursor needs a second reader */
  X(db,"insert-committed","INSERT INTO ft VALUES('apple xx qq segone');"
                          "INSERT INTO ft VALUES('apple segtwo');");
  /* uncommitted, so these stay in the pending-terms hash while the scan runs */
  if(X(db,"begin","BEGIN;")) { sqlite3_close(db); return 1; }
  X(db,"insert-pending","INSERT INTO ft VALUES('apple pendingone');"
                        "INSERT INTO ft VALUES('apple pendingtwo');");

  sqlite3_stmt *st=0;
  if(sqlite3_prepare_v2(db,
        "SELECT offsets(ft) FROM ft WHERE ft MATCH "
        "'(xx AND (apple NEAR/3 qq)) OR pendingone'",-1,&st,0)==SQLITE_OK){
    int n=0, row1ok=0, src, orc=-1;
    while((src=sqlite3_step(st))==SQLITE_ROW){
      const char *v;
      n++;
      out_text("fts3zterm row ");out_uint((unsigned)n);out_text(" offsets");
      v=(const char*)sqlite3_column_text(st,0);
      out_text("=[");out_text(v?v:"NULL");out_text("]\n");
      if(n==1){
        if(streq(v, ROW1_OFFSETS)) row1ok=1;
        /* frees the pending-terms hash under the open cursor */
        out_text("fts3zterm mid-scan optimize\n");
        orc = X(db,"optimize","INSERT INTO ft(ft) VALUES('optimize');");
      }
    }
    sqlite3_finalize(st);
    out_text("fts3zterm rows=");out_uint((unsigned)n);
    out_text(" row1_expected=");out_uint((unsigned)row1ok);
    out_text(" optimize_rc=");out_uint((unsigned)(orc<0?999:orc));
    out_text(" last_step_rc=");out_uint((unsigned)src);out_text("\n");
    if(!row1ok || orc!=0){
      out_text("fts3zterm WARNING row 1 differs or the mid-scan optimize failed -- the"
               " pending-terms path was NOT exercised. Is SQLITE_ENABLE_FTS3_PARENTHESIS"
               " defined for this group?\n");
    }
  } else { out_text("fts3zterm prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }

  X(db,"commit","COMMIT;");
  sqlite3_close(db);
  out_text("fts3zterm NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3zterm")
