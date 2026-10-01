/* rtree group baseline probe -- the analogue of fts5probe / fts3probe.
 * Exercises exactly the machinery the two rtree bug cases share (CREATE VIRTUAL
 * TABLE ... USING rtree, a multi-node insert load, a full scan and a constrained
 * scan) but performs NO nested write, installs NO shadow-table trigger and opens
 * NO second connection.  If this PASSes while rtreecursor/rtreeinode0 FAULT, the
 * faults are attributable to their triggering steps rather than to the group.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void rcline(const char *what, int rc){
  out_text("rtreeprobe "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; char *e=0; int rc;

  rc = sqlite3_open(":memory:", &db);
  rcline("open", rc);
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  rc = sqlite3_exec(db,"PRAGMA page_size=512;",0,0,&e);
  rcline("pagesize", rc); if(e){ sqlite3_free(e); e=0; }

  rc = sqlite3_exec(db,"CREATE VIRTUAL TABLE t1 USING rtree(id,x1,x2);",0,0,&e);
  rcline("create", rc);
  if(e){ out_text("rtreeprobe create err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  sqlite3_stmt *ins=0;
  if(sqlite3_prepare_v2(db,"INSERT INTO t1 VALUES(?,?,?)",-1,&ins,0)!=SQLITE_OK){
    out_text("rtreeprobe insert prepare err ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
    sqlite3_close(db); return 1;
  }
  int i, nins=0;
  for(i=1;i<=30;i++){
    sqlite3_bind_int(ins,1,i); sqlite3_bind_int(ins,2,i); sqlite3_bind_int(ins,3,i+1);
    if(sqlite3_step(ins)==SQLITE_DONE) nins++;
    sqlite3_reset(ins);
  }
  sqlite3_finalize(ins);
  out_text("rtreeprobe inserted="); out_uint((unsigned)nins); out_text("\n");

  /* full scan: drives nodeAcquire/nodeRelease over every node */
  sqlite3_stmt *st=0; int n=0;
  if(sqlite3_prepare_v2(db,"SELECT id,x1 FROM t1",-1,&st,0)==SQLITE_OK){
    while(sqlite3_step(st)==SQLITE_ROW) n++;
    sqlite3_finalize(st); st=0;
  }
  out_text("rtreeprobe scanned="); out_uint((unsigned)n); out_text("\n");

  /* constrained scan: drives findLeafNode/rtreeStepToLeaf */
  n=0;
  if(sqlite3_prepare_v2(db,"SELECT id FROM t1 WHERE x1>5 AND x1<20",-1,&st,0)==SQLITE_OK){
    while(sqlite3_step(st)==SQLITE_ROW) n++;
    sqlite3_finalize(st); st=0;
  }
  out_text("rtreeprobe constrained="); out_uint((unsigned)n); out_text("\n");

  /* a plain DELETE with NO cursor open -- the benign counterpart of rtreecursor */
  rc = sqlite3_exec(db,"DELETE FROM t1 WHERE id>20;",0,0,&e);
  rcline("standalone delete", rc);
  if(e){ out_text("rtreeprobe del err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }

  n=0;
  if(sqlite3_prepare_v2(db,"SELECT count(*) FROM t1",-1,&st,0)==SQLITE_OK){
    if(sqlite3_step(st)==SQLITE_ROW) n=sqlite3_column_int(st,0);
    sqlite3_finalize(st); st=0;
  }
  out_text("rtreeprobe remaining="); out_uint((unsigned)n); out_text("\n");

  sqlite3_close(db);
  out_text("rtreeprobe NOTRAP done\n"); return 0;
}
REPRO322_MAIN("rtreeprobe")
