/* row 1 / sqlite-2639ddc474 -- fts5vocab cursor stepped past EOF.
 *
 * fts5VocabInstanceNext() keeps stepping the fts5 iterator after EOF, dereferencing an
 * iterator whose backing page was released. The 3.26.0 fix adds the EOF test:
 *     if( pCsr->bEof || eDetail==FTS5_DETAIL_NONE ) break;
 *
 * WHAT CHANGED HERE, AND WHY. An earlier version of this case created
 * fts5vocab('ft','row'). The bug is in fts5Vocab**Instance**Next, which only runs for
 * the 'instance' vocab type -- with 'row' (or 'col') a different next-method is used and
 * the buggy function is never entered. That is why the case ran to NOTRAP while
 * establishing nothing. Host ASan on 3.22.0, same setup, one word changed:
 *     fts5vocab('ft','row')      -> 4 rows, silent
 *     fts5vocab('ft','col')      -> 4 rows, silent
 *     fts5vocab('ft','instance') -> 6 rows, heap-use-after-free
 *
 * Host ASan oracle (3.22.0):
 *   READ of size 1
 *     use   sqlite3Fts5PoslistNext64 <- fts5VocabInstanceNext <- fts5VocabNextMethod
 *     free  fts5DataRelease <- fts5SegIterNextPage <- fts5SegIterNext
 *             <- fts5MultiIterNext <- sqlite3Fts5IterNextScan
 *
 * REACHABILITY PROBE: the instance vocab must yield 6 rows for this corpus -- one per
 * (term, doc, offset) instance across the three documents. 'row' yields 4. The domain
 * prints the count and warns if it is not 6, so a wrong vocab type reports itself.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void rcline(const char *what, int rc){
  out_text("fts5vocabeof "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; char *e=0; int rc;

  rc = sqlite3_open(":memory:", &db);
  rcline("open", rc);
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  rc = sqlite3_exec(db,
      "CREATE VIRTUAL TABLE ft USING fts5(a);"
      "INSERT INTO ft VALUES('alpha beta');"
      "INSERT INTO ft VALUES('beta gamma');"
      "INSERT INTO ft VALUES('gamma delta');"
      /* 'instance', not 'row': fts5VocabInstanceNext is the buggy function. */
      "CREATE VIRTUAL TABLE vv USING fts5vocab('ft','instance');", 0,0,&e);
  rcline("setup", rc);
  if(e){ out_text("fts5vocabeof setup err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  /* Step the instance cursor all the way through EOF. */
  sqlite3_stmt *st=0;
  rc = sqlite3_prepare_v2(db,"SELECT term, doc, col, offset FROM vv",-1,&st,0);
  rcline("scan prepare", rc);
  if(rc!=SQLITE_OK){
    out_text("fts5vocabeof prepare err ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
    sqlite3_close(db); return 1;
  }
  int n=0;
  while(sqlite3_step(st)==SQLITE_ROW) n++;
  int endrc = sqlite3_errcode(db);
  sqlite3_finalize(st);

  out_text("fts5vocabeof instance_rows="); out_uint((unsigned)n);
  out_text(" end_rc="); out_uint((unsigned)(endrc<0?-endrc:endrc)); out_text("\n");
  out_text("fts5vocabeof   (host oracle on this corpus: instance=6 rows and it faults;"
           " row=4 rows and is silent)\n");
  if(n!=6)
    out_text("fts5vocabeof WARNING expected 6 instance rows."
             " Is the vocab type really 'instance'? With 'row' or 'col' the buggy"
             " fts5VocabInstanceNext is never entered.\n");

  sqlite3_close(db);
  out_text("fts5vocabeof NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5vocabeof")
