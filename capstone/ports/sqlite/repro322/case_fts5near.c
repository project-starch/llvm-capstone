/* row22 / sqlite-2026-06-08 (b677c5afd4) -- fts5ExprNearIsMatch keeps a lookahead reader
 * on each phrase's poslist buffer and rewrites that SAME buffer through a writer while
 * reading it. The code's own comment argues this is safe because the new poslist is a
 * subset of the old, so nothing unread is overwritten -- but it says nothing about
 * REALLOCATION, and sqlite3Fts5PoslistWriterAppend begins with
 *
 *     if( fts5BufferGrow(&rc, pBuf, 5+5+5) ) return rc;
 *
 * which calls sqlite3Fts5BufferSize and sqlite3_realloc as soon as n+15 > nSpace. The
 * buffer moves, the reader keeps reading the old one. Fixed 3.53.3.
 *
 * WHY THESE FOUR ROWS. nSpace doubles from 64, and the retained poslist is a subset of
 * the original, so n_new <= n_old <= nSpace: the realloc fires only when the retained
 * length lands in the top 15 bytes of an allocation bucket. Measured 2026-10-16 by
 * sweeping one row of "aa bb" repeats under host ASan:
 *
 *     repeats     50  100  118  125  150  200  245  250  255  400  500  505
 *     reports      .    .    X    X    .    .    X    X    X    .    .    X
 *
 * which is exactly the windows (49,64] (113,128] (241,256] (497,512]. The trigger
 * recorded until today had two rows of ten-odd words, a poslist of a few bytes against
 * nSpace 64, so n+15 never exceeded it and no append ever reallocated. One row is not
 * enough to be robust against small encoding differences, so this case indexes four, one
 * per window, and reports which rowids it reached.
 *
 * Host ASan oracle (prepared 3.22.0 amalgamation, system allocator, lookaside off):
 *   READ of size 1, 52 bytes inside a freed 64-byte region
 *     use   sqlite3Fts5PoslistNext64 <- fts5LookaheadReaderNext <- fts5ExprNearIsMatch
 *             <- fts5ExprNearTest <- fts5ExprNodeTest_STRING <- fts5ExprNodeTest
 *     free  realloc <- sqlite3MemRealloc <- sqlite3_realloc <- sqlite3Fts5BufferSize
 *             <- sqlite3Fts5PoslistWriterAppend
 * REACHABILITY: the stale read happens inside the FIRST sqlite3_step, before any row is
 * returned, so the markers come earlier -- "indexed" after the inserts and "scanning"
 * before the first step. A run that reaches "scanning" exercised the path; one that
 * reports rows=0 afterwards did not match and says so.
 * CONTROL: on unprotected capstone the stale read completes -> NOTRAP.
 * NOTE: build in fts5 group. -DSQLITE_DQS=0, so SQL literals are single-quoted. */
#include "repro322_common.h"

/* one length per allocation bucket: (49,64] (113,128] (241,256] (497,512] */
static const int aRepeat[4] = { 60, 125, 250, 505 };
static char zDoc[6 * 512 + 16];   /* "aa bb " is 6 bytes, the longest row repeats 505 */

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  if((rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts5(a);",0,0,0))){
    out_text("fts5near create rc=");out_uint((unsigned)rc);out_text("\n");
    sqlite3_close(db); return 1;
  }

  int i;
  for(i=0;i<4;i++){
    int j, k = 0;
    for(j=0;j<aRepeat[i];j++){            /* "aa bb " repeated, no libc string calls */
      zDoc[k++]='a'; zDoc[k++]='a'; zDoc[k++]=' ';
      zDoc[k++]='b'; zDoc[k++]='b'; zDoc[k++]=' ';
    }
    zDoc[k]=0;
    char *zSql = sqlite3_mprintf("INSERT INTO ft VALUES(%Q);", zDoc);
    if(!zSql){ out_text("fts5near oom\n"); sqlite3_close(db); return 1; }
    rc = sqlite3_exec(db, zSql, 0, 0, 0);
    sqlite3_free(zSql);
    if(rc){ out_text("fts5near insert rc=");out_uint((unsigned)rc);
            out_text(" repeats=");out_uint((unsigned)aRepeat[i]);out_text("\n"); }
  }

  out_text("fts5near indexed=4 repeats=60,125,250,505\n");

  sqlite3_stmt *st=0;
  if(sqlite3_prepare_v2(db,"SELECT rowid FROM ft WHERE ft MATCH 'NEAR(aa bb, 10)'",
                        -1,&st,0)==SQLITE_OK){
    int n=0, src;
    /* the rewrite happens inside this first step, so this marker is the last thing a
     * protected arm prints */
    out_text("fts5near scanning\n");
    while((src=sqlite3_step(st))==SQLITE_ROW){
      n++;
      out_text("fts5near row ");out_uint((unsigned)sqlite3_column_int(st,0));out_text("\n");
    }
    sqlite3_finalize(st);
    out_text("fts5near rows=");out_uint((unsigned)n);
    out_text(" last_step_rc=");out_uint((unsigned)src);out_text("\n");
    if(n==0){
      out_text("fts5near WARNING no row matched -- fts5ExprNearIsMatch never ran, so the"
               " poslist rewrite was not exercised\n");
    }
  } else { out_text("fts5near prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }

  sqlite3_close(db);
  out_text("fts5near NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5near")
