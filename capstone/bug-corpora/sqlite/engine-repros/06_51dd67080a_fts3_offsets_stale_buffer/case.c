/* row13 / sqlite-51dd67080a -- sqlite3Fts3Offsets() saves one poslist pointer per
 * phrase in aTerm[] and then reads through them. A phrase whose poslist is fetched
 * later in the SAME pass restarts the NEAR group it belongs to (fts3EvalRestart,
 * fts3.c:5540), and that frees the buffer an earlier phrase's saved pointer points
 * into. Fixed 3.49.0.
 *
 * WHY THIS QUERY SHAPE. Three facts in the 3.22.0 evaluator pin it down:
 *   1. fts3EvalRestart walks the whole subtree of pNear, and pNear is the phrase's
 *      most senior NEAR ancestor -- so the freed phrase and the saved phrase must be
 *      in ONE NEAR group. In a flat "a OR b" every phrase has pNear = itself: the
 *      restart can only free the poslist of the phrase being fetched, whose pointer
 *      is stored after that fetch returns. That was this case's trigger until
 *      2026-10-16 and it never reached the defect.
 *   2. The restart branch is reached only below an OR, and only for a phrase that is
 *      incremental and not positioned on the current row.
 *   3. fts3EvalTestExpr zeroes the poslists of a NEAR group that does not match the
 *      current row, before any auxiliary function runs, so nothing would be saved --
 *      unless an AND operand to its left fails first, because
 *      bHit = (TestExpr(pLeft) && TestExpr(pRight) && NearTest()) short-circuits and
 *      never descends into the NEAR node.
 * Hence (xx AND ("pp1 pp2" NEAR/3 qq)) OR rr over two rows: at row 2, xx is
 * exhausted (so the cleanup is skipped), "pp1 pp2" is on the row and its pointer is
 * saved, qq is exhausted and incremental so its fetch restarts the group and frees
 * that buffer, and rr is why row 2 is visited at all.
 *
 * Host ASan oracle (prepared 3.22.0 amalgamation, system allocator, lookaside off):
 *   READ of size 1, 1 byte inside a freed 8-byte region
 *     use   sqlite3Fts3Offsets
 *     free  fts3EvalInvalidatePoslist <- fts3EvalRestart <- fts3EvalRestart
 *             <- sqlite3Fts3EvalPhrasePoslist <- fts3ExprTermOffsetInit
 *     alloc fts3EvalIncrPhraseNext -- the merged poslist of the two-token phrase
 *
 * THIS CASE NEEDS ITS OWN GROUP: the nesting and the AND operator exist only with
 * -DSQLITE_ENABLE_FTS3_PARENTHESIS. Without it the parentheses and AND are ordinary
 * tokens, a different query runs, and the case would pass having tested nothing --
 * so it reports the row count and says so.
 * REACHABILITY PROBE, so this can never again pass vacuously: row 1's offsets() must
 * be exactly the five quadruples below -- one per aTerm entry, the two-token phrase
 * contributing two -- which is fixed by the document text and so holds under every
 * allocator. That is the only invariant. The stale read is inside the SECOND row's
 * offsets() call, and what it does to that row is the measurement, not a
 * precondition: on stock memsys5 the read completes, the garbage it returns makes
 * offsets() fail, and the row comes back as an error instead of a row, so rows=1 and
 * last_step_rc is not SQLITE_DONE. Measured 2026-10-16, probe site 45: pp's poslist
 * is saved at base+1, freed as an 8-byte block while qq's fetch restarts the group,
 * and then read at +1 and +2.
 * CONTROL: on unprotected capstone the stale read completes -> NOTRAP.
 * NOTE: build in fts3P group. -DSQLITE_DQS=0, so SQL literals are single-quoted. */
#include "repro322_common.h"

/* row 1 matches every term, so its offsets() is fixed by the document text alone and
 * is the same under every allocator: xx@0, pp1@3, pp2@7, qq@11, rr@14. */
#define ROW1_OFFSETS "0 0 0 2 0 1 3 3 0 2 7 3 0 3 11 2 0 4 14 2"

static int streq(const char *a, const char *b){
  if(!a || !b) return 0;
  while(*a && *a==*b){ a++; b++; }
  return *a==0 && *b==0;
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts3(a);"
    "INSERT INTO ft VALUES('xx pp1 pp2 qq rr');"
    "INSERT INTO ft VALUES('pp1 pp2 rr');",0,0,0);
  if(rc){out_text("fts3offsets setup rc=");out_uint((unsigned)rc);
         out_text(" (");out_text(sqlite3_errmsg(db));out_text(")\n");}
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,
        "SELECT rowid, offsets(ft) FROM ft WHERE ft MATCH "
        "'(xx AND (\"pp1 pp2\" NEAR/3 qq)) OR rr'",-1,&st,0)==SQLITE_OK){
    int n=0, row1ok=0, src;
    while((src=sqlite3_step(st))==SQLITE_ROW){
      const char *v;
      n++;
      /* printed before the value, so the log says which row was being built when a
       * protected arm stopped the program */
      out_text("fts3offsets row ");out_uint((unsigned)sqlite3_column_int(st,0));
      out_text(" offsets");
      v=(const char*)sqlite3_column_text(st,1);
      out_text("=[");out_text(v?v:"NULL");out_text("]\n");
      if(n==1 && streq(v, ROW1_OFFSETS)) row1ok=1;
    }
    sqlite3_finalize(st);
    out_text("fts3offsets rows=");out_uint((unsigned)n);
    out_text(" row1_expected=");out_uint((unsigned)row1ok);
    out_text(" last_step_rc=");out_uint((unsigned)src);
    if(src!=SQLITE_DONE){out_text(" (");out_text(sqlite3_errmsg(db));out_text(")");}
    out_text("\n");
    if(!row1ok){
      out_text("fts3offsets WARNING row 1 is missing or differs -- the NEAR-under-OR"
               " path was NOT taken. Is SQLITE_ENABLE_FTS3_PARENTHESIS defined for"
               " this group?\n");
    }
  } else { out_text("fts3offsets prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts3offsets NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts3offsets")
