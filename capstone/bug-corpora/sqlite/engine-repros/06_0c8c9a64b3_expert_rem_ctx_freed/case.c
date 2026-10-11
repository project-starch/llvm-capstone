/* row19 / sqlite-0c8c9a64b3 -- sqlite3expert idxPopulateStat1() registers rem()/
 * sample() on the USER connection (default iSample=100 -> dbrem = p->db), with a
 * heap IdxRemCtx (rem) and a STACK IdxSampleCtx (sample), then frees the ctx and
 * returns WITHOUT unregistering (sqlite3expert.c:1671/1676/1719). The functions
 * stay registered pointing at freed memory; calling rem() runs idxRemFunc over the
 * freed context. Fixed 3.46.1. CONTROL: run the expert flow, then SELECT rem(...).
 * On unprotected capstone the freed-context read completes -> NOTRAP.
 * Build in the ext group: DOMAIN_EXTRA_SRC includes ext/expert/sqlite3expert.c and
 * -I points at ext/expert for this header. */
#include "repro322_common.h"
#include "sqlite3expert.h"

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,
    "CREATE TABLE t(a INTEGER, b TEXT);"
    "INSERT INTO t VALUES(1,'x'),(2,'y'),(3,'z'),(4,'w');",0,0,0);
  if(rc){out_text("expertrem setup rc=");out_uint((unsigned)rc);out_text(" (");out_text(sqlite3_errmsg(db));out_text(")\n");}

  char *zErr=0;
  sqlite3expert *p = sqlite3_expert_new(db,&zErr);
  if(!p){ out_text("expertrem expert_new failed (");out_text(zErr?zErr:"?");out_text(")\n");
          sqlite3_free(zErr); sqlite3_close(db); return 1; }
  out_text("expertrem expert_new ok\n");

  rc = sqlite3_expert_sql(p,"SELECT * FROM t WHERE a=1 AND b='x';",&zErr);
  out_text("expertrem expert_sql rc=");out_uint((unsigned)(rc<0?-rc:rc));out_text("\n");

  /* registers rem()/sample() on db with a context it then frees */
  rc = sqlite3_expert_analyze(p,&zErr);
  out_text("expertrem analyze rc=");out_uint((unsigned)(rc<0?-rc:rc));
  out_text(" (");out_text(zErr?zErr:"");out_text(")\n");
  sqlite3_free(zErr); zErr=0;

  sqlite3_expert_destroy(p);   /* frees the expert (dbm/dbv); rem()/sample() stay on db */
  out_text("expertrem destroyed; calling rem() over freed ctx\n");

  sqlite3_stmt *st=0;
  if(sqlite3_prepare_v2(db,"SELECT rem(0, a) FROM t",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;      /* idxRemFunc reads freed IdxRemCtx */
    out_text("expertrem rem rows=");out_uint((unsigned)n);out_text("\n");
    sqlite3_finalize(st);
  } else { out_text("expertrem rem prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }

  sqlite3_close(db);
  out_text("expertrem NOTRAP done\n"); return 0;
}
REPRO322_MAIN("expertrem")
