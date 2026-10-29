/* row11 / sqlite-dee0359ddb -- fts3SegReaderNext() points pReader->zTerm straight
 * into the pending-terms hash key memory (fts3_write.c:1333), and
 * sqlite3Fts3SegReaderFree() deliberately does NOT free it for a pending reader
 * (fts3_write.c:1581), so the term is only ever borrowed.
 * sqlite3Fts3PendingTermsClear() frees that hash, and any later read of zTerm is
 * a use-after-free. Fixed 3.37.0 by copying the term into the reader.
 *
 * THE TRIGGER WAS WRONG UNTIL 2026-10-29 AND THIS CASE MEASURED NOTHING.
 * It inserted three rows outside a transaction and ran optimize() mid-scan.
 * Each INSERT auto-commits, so PendingTermsClear had already run three times
 * before the SELECT opened and the hash was empty -- there was no borrowed term
 * to dangle. Keeping the inserts inside BEGIN fixes that half and is still not
 * enough: in 3.22.0 no pending reader that survives across rows ever reads zTerm
 * again. An exact-term reader snapshots ONE hash element plus a NULL sentinel
 * (sqlite3Fts3SegReaderPending), so its next advance sees NULL and stops; a
 * prefix reader snapshots every matching term, but fts3TermSelect consumes all
 * of them eagerly and frees the reader before the first row is returned.
 *
 * What does read a borrowed zTerm after the clear is a RESTART. matchinfo()
 * reaches sqlite3Fts3EvalPhraseStats -> fts3EvalGatherStats -> fts3EvalRestart,
 * which re-starts the segment readers built before the clear, and
 * sqlite3Fts3SegReaderStep then compares their terms. So the two aux functions
 * have to run in this order on the SAME row: optimize() frees the hash (being a
 * write it goes sqlite3VtabSavepoint -> fts3SavepointMethod -> fts3SyncMethod ->
 * PendingTermsFlush -> PendingTermsClear), then matchinfo() restarts over the
 * freed term. Swapping the two columns, or dropping matchinfo(), is silent.
 *
 * CONTROL arm: one row inserted inside an open transaction, then
 * SELECT docid, optimize(ft), matchinfo(ft) over MATCH 'one'. Host ASan reports
 * heap-use-after-free, a 3-byte read of the freed hash key in
 * sqlite3Fts3SegReaderStep under fts3EvalRestart, freed by fts3HashFree under
 * sqlite3Fts3PendingTermsClear.
 * NOTE: build in fts3 group. */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts3(x);",0,0,0);
  if(rc)return FAILRC("fts3zterm create",rc);
  /* no COMMIT: the term stays in the pending-terms hash, so the scan below
   * borrows a pointer into it */
  rc=sqlite3_exec(db,"BEGIN;INSERT INTO ft VALUES('d one');",0,0,0);
  if(rc)return FAILRC("fts3zterm pending",rc);
  sqlite3_stmt*st=0;
  /* column order is load-bearing: optimize() frees the hash, then matchinfo()
   * restarts the readers over the freed term */
  if(sqlite3_prepare_v2(db,"SELECT docid, optimize(ft), matchinfo(ft)"
                           " FROM ft WHERE ft MATCH 'one'",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;
    out_text("fts3zterm rows=");out_uint((unsigned)n);out_text("\n"); sqlite3_finalize(st);
  } else { out_text("fts3zterm prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts3zterm NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3zterm")
